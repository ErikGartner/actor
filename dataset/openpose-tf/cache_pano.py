'''
All code is highly based on Ildoo Kim's code
(https://github.com/ildoonet/tf-openpose)
and derived from the OpenPose Library
(https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
'''
import os
import tensorflow as tf
import cv2
import json
import numpy as np
import argparse
import scipy.misc
import scipy.io

from tqdm import tqdm
from common import estimate_pose, draw_humans, read_imgfile

import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorflow Openpose Inference')
    parser.add_argument('--panopticpath', type=str)
    parser.add_argument('--cachepath', type=str)
    parser.add_argument('--scale-factor', help='Factor to rescale image by before sending into network', type=int, default=4)
    args = parser.parse_args()

    t0 = time.time()

    tf.reset_default_graph()

    from tensorflow.core.framework import graph_pb2
    graph_def = graph_pb2.GraphDef()
    with open('models/optimized_openpose.pb', 'rb') as f:
        graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

    inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
    heatmaps_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L2/BiasAdd:0')
    pafs_tensor = tf.get_default_graph().get_tensor_by_name('Mconv7_stage6_L1/BiasAdd:0')
    base_feat_tensor = tf.get_default_graph().get_tensor_by_name('conv4_4_CPM/BiasAdd:0')

    # List all sub-scenes of the panoptic path
    scene_names = sorted(os.listdir(args.panopticpath))
    with tf.Session() as sess:

        # Iterate over sub-scenes
        for scene_name in tqdm(scene_names):

            # Create scene save path
            save_path_scene = os.path.join(args.cachepath, 'openpose', scene_name)
            try:
                os.mkdir(save_path_scene)
            except:
                pass

            # List all cameras in sub-scene
            scene_path = os.path.join(args.panopticpath, scene_name)
            cam_names = sorted(os.listdir(os.path.join(scene_path, 'hdImgs')))

            # Iterate over cameras
            for cam_name in tqdm(cam_names):

                # List all images (frames) within camera
                cam_path = os.path.join(scene_path, 'hdImgs', cam_name)
                img_names = sorted(os.listdir(cam_path))

                # Containers to save
                heatMats = []
                pafMats = []
                baseFeatMats = []
                joints2d = []

                # Iterate over images
                for img_name in img_names:

                    # Read current image
                    img_path = os.path.join(cam_path, img_name)
                    image, IMG_WIDTH, IMG_HEIGHT = read_imgfile(img_path, args.scale_factor)

                    # Run OpenPose-net on current image
                    heatMat, pafMat, baseFeatMat = \
                        sess.run([heatmaps_tensor, pafs_tensor,
                                  base_feat_tensor],
                                 feed_dict={inputs: image})
                    heatMat = heatMat[0]
                    pafMat = pafMat[0]
                    baseFeatMat = baseFeatMat[0]

                    # Compute 2d pose based on OpenPose-net output
                    humans = estimate_pose(heatMat, pafMat)

                    # Go from [0,1]-normalized joint coordinates to instead
                    # match image size
                    humans_formatted = []
                    for i in range(len(humans)):
                        human = humans[i]
                        human_formatted = []
                        for key, val in human.items():
                            hum = list(human[key])
                            # human[key] = list(human[key])
                            hum[1] = list(human[key][1])
                            hum[1][0] *= IMG_WIDTH
                            hum[1][1] *= IMG_HEIGHT
                            tmp = {'id': hum[0], 'xy': [float(xy) for xy in hum[1]], 'conf': float(hum[2])}
                            human_formatted.append(tmp)
                        humans_formatted.append(human_formatted)

                    # Append to containers
                    baseFeatMats.append(baseFeatMat)
                    joints2d.append(humans_formatted)

                # Save all to file (to the cache path)
                baseFeatMats = np.asarray(baseFeatMats)
                save_path_cam = os.path.join(save_path_scene, cam_name)
                os.makedirs(save_path_cam)
                save_path = os.path.join(save_path_cam, 'conv4_4_CPM.mat')
                scipy.io.savemat(save_path, {'conv4_4_CPM': baseFeatMats})

                save_path = os.path.join(save_path_cam, 'joints2d.json')
                with open(save_path, 'w') as f:
                    json.dump(joints2d, f, indent=2)
