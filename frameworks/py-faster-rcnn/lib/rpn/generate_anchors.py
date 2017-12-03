#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# Modified by Håvard Thom
# --------------------------------------------------------

import numpy as np
from configs.config import cfg
import os.path as osp


def load_custom_anchors():
    """
    Load custom anchor points for dataset
    """
    anchor_file = osp.join(cfg.OUTPUT_DIR, '9_anchor_boxes.txt')

    # Read anchor file
    with open(anchor_file, 'r') as f:
        data = f.readlines()

    nb_anchors = len(data)-1

    custom_anchors = np.zeros((nb_anchors, 4), dtype=np.float32)

    for i in range(0, nb_anchors):
        splt = data[i+1].split(',') # Skip first line header

        # Scale anchors to training image size
        anchor_width = float(splt[0])*cfg.TRAIN.MAX_SIZE
        anchor_height = float(splt[1])*cfg.TRAIN.SCALES[0]

        # convert from [x, y, w, h] to [xmin, ymin, xmax, ymax]
        xmin = -anchor_width/2.0
        ymin = -anchor_height/2.0
        xmax = anchor_width/2.0
        ymax = anchor_height/2.0

        custom_anchors[i][0] = xmin
        custom_anchors[i][1] = ymin
        custom_anchors[i][2] = xmax
        custom_anchors[i][3] = ymax

    return custom_anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    # a = generate_custom_anchors()
    print time.time() - t
    print a
    # from IPython import embed; embed()
