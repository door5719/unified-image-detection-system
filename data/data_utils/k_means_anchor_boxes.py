#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Create custom anchor boxes for a bounding box dataset using k-means clustering"""
from __future__ import print_function
import sys
import argparse
import numpy as np
# import pandas as pd
import os.path as osp
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import FuncFormatter

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from utils import make_if_not_exist, check_if_exist

INFO = False

def overlap(x1, w1, x2, w2):
    """Calculate overlap of two bounding boxes"""
    l1 = x1 - w1/2.
    l2 = x2 - w2/2.
    left = l1 if l1 > l2 else l2
    r1 = x1 + w1/2.
    r2 = x2 + w2/2.
    right = r1 if r1 < r2 else r2
    return right - left

def intersection(bbox1, bbox2):
    """Calculate intersection of two bounding boxes"""
    w = overlap(bbox1[0], bbox1[2], bbox2[0], bbox2[2])
    h = overlap(bbox1[1], bbox1[3], bbox2[1], bbox2[3])
    if w < 0 or h < 0:
        return 0
    return w*h

def area(bbox):
    """Calculate area of a bounding box"""
    return bbox[2]*bbox[3]

def union(bbox1, bbox2):
    """Calculate union of two bounding boxes"""
    return area(bbox1) + area(bbox2) - intersection(bbox1, bbox2)

def iou(bbox1, bbox2):
    """Calculate intersection over union of two bounding boxes"""
    return intersection(bbox1, bbox2) / union(bbox1, bbox2)

def niou(bbox1, bbox2):
    """Calculate inverse intersection over union of two bounding boxes"""
    return 1. - iou(bbox1, bbox2)

def find_closest_centroid(bbox, centroids):
    """Find closest centroid to a bounding box"""
    min_distance = float('inf')
    cluster_idx = None
    for i, centroid in enumerate(centroids):
        distance = niou(bbox, centroid)

        if distance < min_distance:
            min_distance = distance
            cluster_idx = i

    return cluster_idx, min_distance

def k_means(k, centroids, bboxes, iteration_cutoff=25):
    """
    Finds best clusters for bounding box data with k-means,
    distance measure is inverse IoU, therefore minimizing distance = maximizing IoU.
    """

    anchor_boxes = []
    best_avg_iou = 0
    best_avg_iou_iteration = 0
    iter_count = 0

    while True:
        clusters = [[] for _ in range(k)]
        clusters_iou = []

        for bbox in bboxes:
            # Calculate minimum distance from bbox to current centroids
            cluster_idx, distance = find_closest_centroid(bbox, centroids)
            # Add bbox to appropriate cluster
            clusters[cluster_idx].append(bbox)
            # Save distance
            clusters_iou.append(1. - distance)

        # Calculate new centroids
        new_centroids = [np.mean(cluster, axis=0) for cluster in clusters]

        # Average IoU this iteration
        avg_iou = np.mean(clusters_iou)

        # Save best centroids as anchor_boxes
        if avg_iou > best_avg_iou:
            best_avg_iou = avg_iou
            best_avg_iou_iteration = iter_count
            anchor_boxes = new_centroids

        if INFO:
            # Print iteration stats
            print('Iteration {:d}'.format(iter_count))
            print('Average iou to closest centroid = {:f}'.format(avg_iou))
            # print('Sum of all distances (cost) = {:f}\n'.format(np.sum(clusters_niou)))

            # Print cluster and centroid stats
            for i in range(len(new_centroids)):
                shift = niou(centroids[i], new_centroids[i])
                print('Cluster {:d} size: {:d}'.format(i, len(clusters[i])))
                print('Centroid {:d} distance shift: {:f}\n\n'.format(i, shift))

        if iter_count >= best_avg_iou_iteration + iteration_cutoff:
            break

        centroids = new_centroids
        iter_count+=1

    return anchor_boxes, best_avg_iou

# Original function: https://github.com/scikit-learn/scikit-learn/blob/14031f6/sklearn/cluster/k_means_.py#L45
# Modified by Håvard Thom
def k_init(k, bboxes, n_local_trials=None):
    """Init k cluster centroids according to k-means++"""

    centroids = np.empty((k, bboxes.shape[1]), dtype=bboxes.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        n_local_trials = 2 + int(np.log(k))

    # Pick first center randomly
    if INFO:
        print('k_init: Picking centroid 0 randomly\n')
    center_id = np.random.randint(len(bboxes))
    centroids[0] = bboxes[center_id]

    # Initialize list of closest distances and calculate current cost
    clusters_niou = []
    for bbox in bboxes:
        clusters_niou.append(niou(bbox, centroids[0]))

    clusters_niou = np.asarray(clusters_niou)

    current_cost =  clusters_niou.sum()

    # Pick the remaining k-1 centroids
    for c in range(1, k):
        if INFO:
            print('k_init: Picking centroid {:d}. \
            Current cost: {:f}'.format(c, current_cost))
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = np.random.random_sample(n_local_trials) * current_cost
        candidate_ids = np.searchsorted(clusters_niou.cumsum(), rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = []
        for cand in bboxes[candidate_ids]:
            cand_distances = []
            for bbox in bboxes:
                cand_distances.append(niou(bbox, cand))
            distance_to_candidates.append(cand_distances)

        distance_to_candidates = np.asarray(distance_to_candidates)

        # Decide which candidate is the best
        best_candidate = None
        best_cost = None
        best_dist = None
        for trial in range(n_local_trials):
            # Compute cost when including center candidate
            new_dist = np.minimum(clusters_niou, distance_to_candidates[trial])
            new_cost = new_dist.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_cost < best_cost):
                best_candidate = candidate_ids[trial]
                best_cost = new_cost
                best_dist = new_dist

        # Permanently add best center candidate found in local tries
        centroids[c] = bboxes[best_candidate]

        current_cost = best_cost
        clusters_niou = best_dist
        if INFO:
            print('k_init: Centroid {:d} picked. \
            Current cost: {:f}\n'.format(c, current_cost))

    return centroids

def plot_anchors(anchors1, anchors2):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.set_ylim([0,500])
    ax1.set_xlim([0,900])

    for i in range(len(anchors1)):
        # Order boxes so smallest is in the front
        if area(anchors1[i]) > area(anchors2[i]):
            bbox1 = anchors1[i]
            color1 = "white"
            bbox2 = anchors2[i]
            color2 = "blue"
        else:
            bbox1 = anchors2[i]
            color1 = "blue"
            bbox2 = anchors1[i]
            color2 = "white"

        lower_left_x = bbox1[0]-(bbox1[2]/2.0)
        lower_left_y = bbox1[1]-(bbox1[3]/2.0)

        ax1.add_patch(
            patches.Rectangle(
                (lower_left_x, lower_left_y),   # (x,y)
                bbox1[2],          # width
                bbox1[3],          # height
                facecolor=color1
            ))

        lower_left_x = bbox2[0]-(bbox2[2]/2.0)
        lower_left_y = bbox2[1]-(bbox2[3]/2.0)

        ax1.add_patch(
            patches.Rectangle(
                (lower_left_x, lower_left_y),   # (x,y)
                bbox2[2],          # width
                bbox2[3],          # height
                facecolor=color2
            ))
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])
    plt.show()

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
    description='Create custom anchor boxes for dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset_dir',
                        help="The dataset directory which contains 'images', \
                        'yolov2_Annotations' and 'yolov2_ImageSets' folders.\
                         (Example: ~/datasets/baitcam)", type=str)
    parser.add_argument('--k', default = 5, type = int,
                        help='Number of anchor boxes (cluster centroids) to generate.')
    parser.add_argument('--info',
            help='Show cluster stats, centroid stats and avg. IoU for each iteration',
            action='store_true', default=False)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    print('Called with args:')
    print(args)

    dataset_dir = args.dataset_dir
    k = args.k
    INFO = args.info

    # Remove trailing slash
    dataset_dir = dataset_dir.rstrip('/')

    yolov2_annotations_dir = osp.join(dataset_dir, 'yolov2_Annotations')
    yolov2_imageset_dir = osp.join(dataset_dir, 'yolov2_ImageSets')

    check_if_exist('Dataset directory', dataset_dir)
    check_if_exist('Annotation directory', yolov2_annotations_dir)
    check_if_exist('Imageset directory', yolov2_imageset_dir)

    train_imageset = osp.join(yolov2_imageset_dir, 'train.txt')
    bb_data = []
    extensions = ['.png', '.jpg', '.JPEG', '.JPG']

    # Load dataset bounding box annotation data (yolov2 format)
    # shape: [[x1,y1,w1,h1],...,[xn,yn,wn,hn]]
    with open(train_imageset, 'r') as f:
        for line in f:
            line = line.replace('images', 'yolov2_Annotations')
            for ext in extensions:
                if ext in line:
                    line = line.replace(ext, '.txt').strip()

            with open(line, 'r') as l:
                for line2 in l:
                    bbox_list = line2.split(' ')[1:]
                    bbox_list = [float(x.strip()) for x in bbox_list]
                    bb_data.append(bbox_list)


    # bb_data = sorted(bb_data)
    # np.random.shuffle(bb_data)
    bb_data = np.array(bb_data)

    # Set x,y coordinates to origin, only need width and height for clustering
    for i in range(len(bb_data)):
        bb_data[i][0] = 0
        bb_data[i][1] = 0

    # centroids = bb_data[:k]
    centroids = k_init(k, bb_data)
    # centroids = k_init(k, bb_data, n_local_trials=len(bb_data))

    # Start k-means to find best clusters
    anchor_boxes, best_avg_iou = k_means(k, centroids, bb_data)

    # Sort on width
    anchor_boxes = np.asarray(anchor_boxes)
    anchor_boxes = anchor_boxes[anchor_boxes[:, 2].argsort()]

    anchor_dir = osp.join(dataset_dir, 'custom_anchor_boxes')
    make_if_not_exist(anchor_dir)

    anchor_file = osp.join(anchor_dir, '{:d}_anchor_boxes.txt'.format(k))

    # Check that better anchor_boxes does not already exist and write to anchor file
    old_iou = 0
    if osp.exists(anchor_file):
        with open(anchor_file, 'r') as f:
            header = f.readline()
            old_iou = float(header.split(':')[1])

    if old_iou > 0:
        print('Previous IoU: {:f}. New IoU: {:f}'.format(old_iou, best_avg_iou))

    if best_avg_iou > old_iou:
        print('Writing anchor boxes to {:s}'.format(anchor_file))
        with open(anchor_file, 'w') as f:
            print('Best average IoU: {:f}'.format(best_avg_iou), file=f)
            for a in anchor_boxes:
                print('{:f},{:f}'.format(a[2], a[3]), file=f)
    else:
        print('Anchor boxes with higher avg. IoU already exists in {:s}'.format(anchor_file))

    # # ### PLOTTING YOLO AND CUSTOM ANCHORS FOR THESIS
    # faster_rcnn_anchors = np.asarray(
    # [[0., 0.,  183.,   95.],
    #  [0., 0.,  367.,  191.],
    #  [0., 0.,  735.,  383.],
    #  [0., 0.,  127.,  127.],
    #  [0., 0.,  255.,  255.],
    #  [0., 0.,  511.,  511.],
    #  [0., 0.,  87.,   175.],
    #  [0., 0.,  175.,  351.],
    #  [0., 0.,  351.,  703.]])
    #
    # # Scale anchors to our image size and normalize
    # scale_w = 2043/833
    # scale_h = 1472/600
    # faster_rcnn_anchors[:,2] *= scale_w
    # faster_rcnn_anchors[:,3] *= scale_h
    # faster_rcnn_anchors[:,2] /= 2043
    # faster_rcnn_anchors[:,3] /= 1472
    #
    # clusters_iou = []
    # for bbox in bb_data:
    #     idx, dist = find_closest_centroid(bbox, faster_rcnn_anchors)
    #     clusters_iou.append(1.-dist)
    # print('Faster R-CNN average IoU: {:.2f}'.format(np.mean(clusters_iou)*100))
    #
    # yolov2_anchors = np.asarray(
    # [[0., 0.,  1.3221,   1.73145],
    #  [0., 0.,  3.19275,  4.00944],
    #  [0., 0.,  5.05587,  8.09892],
    #  [0., 0.,  9.47112,  4.84053],
    #  [0., 0.,  11.2364,  10.0071]])
    #
    # # Scale anchors to our image size and normalize
    # scale_w = 2043/416
    # scale_h = 1472/416
    # yolov2_anchors[:,2] *= scale_w
    # yolov2_anchors[:,3] *= scale_h
    # yolov2_anchors[:,2] /= (2043/32)
    # yolov2_anchors[:,3] /= (1472/32)
    #
    # clusters_iou = []
    # for bbox in bb_data:
    #     idx, dist = find_closest_centroid(bbox, yolov2_anchors)
    #     clusters_iou.append(1.-dist)
    # print('Yolo anchors average IoU: {:.2f}'.format(np.mean(clusters_iou)*100))
    #
    # anchor_file = osp.join(dataset_dir, 'custom_anchor_boxes', '5_anchor_boxes.txt')
    #
    # # Read anchor file
    # with open(anchor_file, 'r') as f:
    #     data = f.readlines()
    #
    # nb_anchors = len(data)-1
    #
    # custom_anchors = np.zeros((nb_anchors, 4), dtype=np.float32)
    #
    # for i in range(0, nb_anchors):
    #     splt = data[i+1].split(',') # Skip first line header
    #     anchor_width = float(splt[0])
    #     anchor_height = float(splt[1])
    #     custom_anchors[i][2] = anchor_width
    #     custom_anchors[i][3] = anchor_height
    #
    # print(custom_anchors)
    # #
    # # custom_anchors = np.asarray(
    # # [[0., 0.,  0.032067, 0.037577],
    # #  [0., 0.,  0.083008, 0.084264],
    # #  [0., 0.,  0.128544, 0.155342],
    # #  [0., 0.,  0.214303, 0.199906],
    # #  [0., 0.,  0.316197, 0.332526]])
    #
    # yolov2_anchors[:,2] *= 2043*0.3
    # yolov2_anchors[:,3] *= 1472*0.3
    #
    # custom_anchors[:,2] *= 2043*0.3
    # custom_anchors[:,3] *= 1472*0.3
    #
    # # Hardcode anchor center coordinates for plot
    # custom_anchors[3][0] = 210
    # custom_anchors[3][1] = 100
    # custom_anchors[0][0] = 320
    # custom_anchors[0][1] = 385
    # custom_anchors[4][0] = 650
    # custom_anchors[4][1] = 175
    # custom_anchors[2][0] = 125
    # custom_anchors[2][1] = 325
    # custom_anchors[1][0] = 320
    # custom_anchors[1][1] = 265
    #
    # yolov2_anchors[3][0] = 210
    # yolov2_anchors[3][1] = 100
    # yolov2_anchors[0][0] = 320
    # yolov2_anchors[0][1] = 385
    # yolov2_anchors[4][0] = 650
    # yolov2_anchors[4][1] = 175
    # yolov2_anchors[2][0] = 125
    # yolov2_anchors[2][1] = 325
    # yolov2_anchors[1][0] = 320
    # yolov2_anchors[1][1] = 265
    #
    # plot_anchors(custom_anchors, yolov2_anchors)
    #
    # # ### AVG IOU GRAPH FOR THESIS
    # all_k = np.arange(1,10)
    # # print(all_k)
    # avg_ious = [0]
    # for k in all_k:
    #     centroids = k_init(k, bb_data)
    #
    #     anchors, best_avg_iou = k_means(k, centroids, bb_data, iteration_cutoff=25)
    #
    #     avg_ious.append(best_avg_iou)
    #
    #     # print(avg_ious)
    #
    # df = pd.DataFrame(avg_ious, columns=['Custom'])
    #
    # ax = df.plot(marker='o')
    #
    # ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
    # ax.set_ylim([0,1])
    # ax.set_xlim([1,10])
    # ax.set_xlabel("Number of anchor boxes", family='Open Sans')
    # ax.set_ylabel("Average IoU", family='Open Sans')
    #
    # # Hardcode YOLO and Faster R-CNN avg IOU from results printed in plotting above
    # ax.plot(5, 0.5658, 'ob')
    # ax.plot(9, 0.5950, 'ob')
    #
    # plt.annotate('YOLO',
    #          xy=(5, 0.5658),
    #          xycoords='data',
    #          textcoords='offset points', family='Open Sans', fontsize=13)
    # plt.annotate('Faster R-CNN',
    #          xy=(9, 0.5950),
    #          xycoords='data',
    #          textcoords='offset points',family='Open Sans', fontsize=13)
    # plt.show()
