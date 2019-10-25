'''
All code is highly based on Ildoo Kim's code (https://github.com/ildoonet/tf-openpose)
and derived from the OpenPose Library (https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/LICENSE)
'''

from collections import defaultdict
from enum import Enum
import math
import time

import numpy as np
import itertools
import cv2
from scipy.ndimage.filters import maximum_filter

import matplotlib.pyplot as plt


class CocoPart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18

CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]   # = 19
CocoPairsRender = CocoPairs[:-2]
CocoPairsNetwork = [
    (12, 13), (20, 21), (14, 15), (16, 17), (22, 23), (24, 25), (0, 1), (2, 3), (4, 5),
    (6, 7), (8, 9), (10, 11), (28, 29), (30, 31), (34, 35), (32, 33), (36, 37), (18, 19), (26, 27)
 ]  # = 19

CocoColors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]


NMS_Threshold = 0.1
InterMinAbove_Threshold = 6
Inter_Threashold = 0.1
Min_Subset_Cnt = 4
Min_Subset_Score = 0.8
Max_Human = 96


def human_conns_to_human_parts(human_conns, heatMat):
    human_parts = defaultdict(lambda: None)
    for conn in human_conns:
        human_parts[conn['partIdx'][0]] = (
            conn['partIdx'][0], # part index
            (conn['coord_p1'][0] / heatMat.shape[2], conn['coord_p1'][1] / heatMat.shape[1]), # relative coordinates
            heatMat[conn['partIdx'][0], conn['coord_p1'][1], conn['coord_p1'][0]] # score
            )
        human_parts[conn['partIdx'][1]] = (
            conn['partIdx'][1],
            (conn['coord_p2'][0] / heatMat.shape[2], conn['coord_p2'][1] / heatMat.shape[1]),
            heatMat[conn['partIdx'][1], conn['coord_p2'][1], conn['coord_p2'][0]]
            )
    return human_parts


def non_max_suppression(heatmap, window_size=3, threshold=NMS_Threshold):
    heatmap[heatmap < threshold] = 0 # set low values to 0
    part_candidates = heatmap*(heatmap == maximum_filter(heatmap, footprint=np.ones((window_size, window_size))))
    return part_candidates


def estimate_pose(heatMat, pafMat):
    if heatMat.shape[2] == 19:
        # transform from [height, width, n_parts] to [n_parts, height, width]
        heatMat = np.rollaxis(heatMat, 2, 0)
    if pafMat.shape[2] == 38:
        # transform from [height, width, 2*n_pairs] to [2*n_pairs, height, width]
        pafMat = np.rollaxis(pafMat, 2, 0)

    # print(np.mean(heatMat, axis=0).shape)
    # plt.figure(1)
    # plt.imshow(np.mean(heatMat, axis=0))
    # plt.figure(2)
    # plt.imshow(np.mean(pafMat, axis=0))
    # plt.show()

    # reliability issue.
    heatMat = heatMat - heatMat.min(axis=1).min(axis=1).reshape(19, 1, 1)
    heatMat = heatMat - heatMat.min(axis=2).reshape(19, heatMat.shape[1], 1)

    _NMS_Threshold = max(np.average(heatMat) * 4.0, NMS_Threshold)
    _NMS_Threshold = min(_NMS_Threshold, 0.3)

    coords = [] # for each part index, it stores coordinates of candidates
    for heatmap in heatMat[:-1]: # remove background
        part_candidates = non_max_suppression(heatmap, 5, _NMS_Threshold)
        coords.append(np.where(part_candidates >= _NMS_Threshold))

    connection_all = [] # all connections detected. no information about what humans they belong to
    for (idx1, idx2), (paf_x_idx, paf_y_idx) in zip(CocoPairs, CocoPairsNetwork):
        connection = estimate_pose_pair(coords, idx1, idx2, pafMat[paf_x_idx], pafMat[paf_y_idx])
        connection_all.extend(connection)

    conns_by_human = dict()
    for idx, c in enumerate(connection_all):
        conns_by_human['human_%d' % idx] = [c] # at first, all connections belong to different humans

    no_merge_cache = defaultdict(list)
    empty_set = set()
    while True:
        is_merged = False
        for h1, h2 in itertools.combinations(conns_by_human.keys(), 2):
            if h1 == h2:
                continue
            if h2 in no_merge_cache[h1]:
                continue
            for c1, c2 in itertools.product(conns_by_human[h1], conns_by_human[h2]):
                # if two humans share a part (same part idx and coordinates), merge those humans
                if set(c1['uPartIdx']) & set(c2['uPartIdx']) != empty_set:
                    is_merged = True
                    # extend human1 connectios with human2 connections
                    conns_by_human[h1].extend(conns_by_human[h2])
                    conns_by_human.pop(h2) # delete human2
                    break
            if is_merged:
                no_merge_cache.pop(h1, None)
                break
            else:
                no_merge_cache[h1].append(h2)

        if not is_merged: # if no more mergings are possible, then break
            break

    # reject by subset count
    conns_by_human = {h: conns for (h, conns) in conns_by_human.items() if len(conns) >= Min_Subset_Cnt}
    # reject by subset max score
    conns_by_human = {h: conns for (h, conns) in conns_by_human.items() if max([conn['score'] for conn in conns]) >= Min_Subset_Score}

    # list of humans
    humans = [human_conns_to_human_parts(human_conns, heatMat) for human_conns in conns_by_human.values()]
    return humans


def estimate_pose_pair(coords, partIdx1, partIdx2, pafMatX, pafMatY):
    connection_temp = [] # all possible connections
    peak_coord1, peak_coord2 = coords[partIdx1], coords[partIdx2]

    for idx1, (y1, x1) in enumerate(zip(peak_coord1[0], peak_coord1[1])):
        for idx2, (y2, x2) in enumerate(zip(peak_coord2[0], peak_coord2[1])):
            score, count = get_score(x1, y1, x2, y2, pafMatX, pafMatY)
            if (partIdx1, partIdx2) in [(2, 3), (3, 4), (5, 6), (6, 7)]: # arms
                if count < InterMinAbove_Threshold // 2 or score <= 0.0:
                    continue
            elif count < InterMinAbove_Threshold or score <= 0.0:
                continue
            connection_temp.append({
                'score': score,
                'coord_p1': (x1, y1),
                'coord_p2': (x2, y2),
                'idx': (idx1, idx2), # connection candidate identifier
                'partIdx': (partIdx1, partIdx2),
                'uPartIdx': ('{}-{}-{}'.format(x1, y1, partIdx1), '{}-{}-{}'.format(x2, y2, partIdx2))
            })

    connection = []
    used_idx1, used_idx2 = [], []
    # sort possible connections by score, from maximum to minimum
    for conn_candidate in sorted(connection_temp, key=lambda x: x['score'], reverse=True):
        # check not connected
        if conn_candidate['idx'][0] in used_idx1 or conn_candidate['idx'][1] in used_idx2:
            continue
        connection.append(conn_candidate)
        used_idx1.append(conn_candidate['idx'][0])
        used_idx2.append(conn_candidate['idx'][1])

    return connection


def get_score(x1, y1, x2, y2, pafMatX, pafMatY):
    num_inter = 10
    dx, dy = x2 - x1, y2 - y1
    normVec = math.sqrt(dx ** 2 + dy ** 2)

    if normVec < 1e-4:
        return 0.0, 0

    vx, vy = dx / normVec, dy / normVec

    xs = np.arange(x1, x2, dx / num_inter) if x1 != x2 else np.full((num_inter, ), x1)
    ys = np.arange(y1, y2, dy / num_inter) if y1 != y2 else np.full((num_inter, ), y1)
    xs = (xs + 0.5).astype(np.int8)
    ys = (ys + 0.5).astype(np.int8)

    # without vectorization
    pafXs = np.zeros(num_inter)
    pafYs = np.zeros(num_inter)
    for idx, (mx, my) in enumerate(zip(xs, ys)):
        pafXs[idx] = pafMatX[my][mx]
        pafYs[idx] = pafMatY[my][mx]

    # vectorization slow?
    # pafXs = pafMatX[ys, xs]
    # pafYs = pafMatY[ys, xs]

    local_scores = pafXs * vx + pafYs * vy
    thidxs = local_scores > Inter_Threashold

    return sum(local_scores * thidxs), sum(thidxs)


def read_imgfile(path, scale_factor):
    img = cv2.imread(path)
    height, width, _ = img.shape
    val_img = preprocess(img, int(width / scale_factor), int(height / scale_factor))
    return val_img, width, height


def preprocess(img, width, height):
    val_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # cv2 reads in BGR format
    val_img = cv2.resize(val_img, (width, height)) # each net accept only a certain size
    val_img = val_img.reshape([1, height, width, 3])
    val_img = val_img.astype(float)
    val_img = val_img * (2.0 / 255.0) - 1.0 # image range from -1 to +1
    return val_img


def draw_humans(img, human_list):
    img_copied = np.copy(img)
    image_h, image_w = img_copied.shape[:2]
    centers = {}
    for human in human_list:
        part_idxs = human.keys()

        # draw point
        for i in range(CocoPart.Background.value):
            if i not in part_idxs:
                continue
            part_coord = human[i][1]
            center = (int(part_coord[0] * image_w + 0.5), int(part_coord[1] * image_h + 0.5))
            centers[i] = center
            cv2.circle(img_copied, center, 3, CocoColors[i], thickness=3, lineType=8, shift=0)

        # draw line
        for pair_order, pair in enumerate(CocoPairsRender):
            if pair[0] not in part_idxs or pair[1] not in part_idxs:
                continue

            img_copied = cv2.line(img_copied, centers[pair[0]], centers[pair[1]], CocoColors[pair_order], 3)

    return img_copied
