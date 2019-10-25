# From Python
# It requires OpenCV installed for Python and OpenPose built with Python API support.
# Tested for OpenPose ~1.4.0
import argparse
import glob
import json
import os
import sys
import time
from sys import platform

import cv2

import matplotlib.pyplot as plt
from scipy.io import savemat
from tqdm import tqdm
import numpy as np


def to_panoptic_format(joints):
    bodies = []
    if len(joints["body"].shape) == 0:
        return bodies

    for pose_id in range(joints["body"].shape[0]):
        body = {
            "id": pose_id,
            "joints25": joints["body"][pose_id, :, :].flatten().tolist(),
            "face": joints["face"][pose_id, :, :].flatten().tolist(),
            "left_hand": joints["left_hand"][pose_id, :, :].flatten().tolist(),
            "right_hand": joints["right_hand"][pose_id, :, :]
            .flatten()
            .tolist(),
        }
        bodies.append(body)
    return bodies


def predict_joints(opWrapper, image):
    datum = op.Datum()
    imageToProcess = cv2.imread(image)
    if imageToProcess is None:
        print(image)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop([datum])

    if (
        len(datum.poseKeypoints.shape) == 0
        and len(datum.faceKeypoints.shape) == 0
        and len(datum.handKeypoints[0].shape) == 0
        and len(datum.handKeypoints[1].shape) == 0
    ):
        return {
            "body": np.array([]),
            "face": np.array([]),
            "left_hand": np.array([]),
            "right_hand": np.array([]),
        }

    #    print("Body keypoints: \n" + str(datum.poseKeypoints))
    #    print("Face keypoints: \n" + str(datum.faceKeypoints))
    #    print("Left hand keypoints: \n" + str(datum.handKeypoints[0]))
    #    print("Right hand keypoints: \n" + str(datum.handKeypoints[1]))
    #
    return {
        "body": datum.poseKeypoints,
        "face": datum.faceKeypoints,
        "left_hand": datum.handKeypoints[0],
        "right_hand": datum.handKeypoints[1],
    }


def main(args):
    params = dict()
    params["model_folder"] = os.path.join(args.openpose_path, "models/")
    params["face"] = True
    params["hand"] = True

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    scenes = sorted(glob.glob(os.path.join(args.dataset_path, "*")))
    for path in tqdm(scenes, desc="scene"):
        cams = sorted(glob.glob(os.path.join(path, "hdImgs", "*")))

        for cam in tqdm(cams, desc="camera"):
            scene_detections = []

            frames = sorted(glob.glob(os.path.join(cam, "*")))

            save_path = os.path.join(
                args.cache_path,
                "openpose",
                os.path.basename(path),
                os.path.basename(cam),
                "joints2d_full",
            )
            save_path_mat = save_path + ".mat"
            save_path_json = save_path + ".json"
            if (
                os.path.exists(save_path_mat)
                and os.path.exists(save_path_json)
                and not args.overwrite
            ):
                print("Already computed. Skipping!")
                continue

            cam_detections = []
            for frame in tqdm(frames, desc="frame"):
                image = frame
                joints = predict_joints(opWrapper, image)

                cam_detections.append(joints)
            scene_detections.append(cam_detections)

            # print("savepath", save_path)
            savemat(save_path_mat, {"poses": scene_detections})

            # print("savepath", save_path)
            with open(save_path_json, "w") as f:
                json.dump(
                    [
                        [to_panoptic_format(x) for x in z]
                        for z in scene_detections
                    ],
                    f,
                    indent=2,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", help="Path to dataset folder")
    parser.add_argument(
        "cache_path", help="Path to cache folder to save detections to"
    )
    parser.add_argument(
        "--openpose_path", help="Path to openpose", default="openpose"
    )
    parser.add_argument(
        "--overwrite", help="Overwrite existing cache", action="store_true"
    )
    args = parser.parse_args()

    try:
        # Import open pose
        sys.path.append(os.path.join(args.openpose_path, "build", "python"))
        from openpose import pyopenpose as op
    except ImportError as e:
        print(
            "Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?"
        )
        raise e

    main(args)
