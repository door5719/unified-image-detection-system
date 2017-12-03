#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Train, test and detect with a Faster R-CNN network."""
from __future__ import print_function
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.test import test_net, im_detect
from fast_rcnn.nms_wrapper import nms
from fast_rcnn_utils.timer import Timer
from configs.config import cfg
from datasets.factory import get_imdb

import caffe
import numpy as np
import sys
import os.path as osp
import cv2
from shutil import copy

from utils import make_if_not_exist, check_if_exist, get_model_path

def train_faster_rcnn(no_pretrained, max_iters):
    """Train a Faster R-CNN network on a region of interest database."""
    # Set pretrained model
    if no_pretrained:
        pretrained_model = None
    else:
        pretrained_model = osp.join(cfg.DATA_DIR, 'imagenet_models',
                           '{:s}.caffemodel'.format(cfg.MODEL_NAME))
        check_if_exist('Pretrained model', pretrained_model)

    # Change solver if OHEM is used
    postfix = ''
    if cfg.TRAIN.USE_OHEM:
        if cfg.MODEL_NAME != 'VGG16' and \
           cfg.MODEL_NAME != 'ResNet101_bn-scale-merged':

            print('Faster RCNN framework with OHEM does not currently '
                  'support model: {:s} (supported models: VGG16, '
                  'ResNet101_bn-scale-merged).').format(cfg.MODEL_NAME)
            sys.exit()
        else:
            postfix = '_ohem'

    # Check if custom anchors exist and copy them to output dir
    if cfg.CUSTOM_ANCHORS:
        anchor_file = osp.join(cfg.DATA_DIR, cfg.DATASET_NAME,
                                   'custom_anchor_boxes', '9_anchor_boxes.txt')
        if not osp.exists(anchor_file):
            print('Custom anchor boxes `{:s}` does not exist.'.format(anchor_file))
            print('Generate custom anchor boxes with '
                  'data/data_utils/k_means_anchor_boxes.py')
            sys.exit()

        copy(anchor_file, osp.join(cfg.OUTPUT_DIR, '9_anchor_boxes.txt'))

    # Set solver
    solver = osp.join(cfg.MODELS_DIR, cfg.DATASET_NAME, cfg.METHOD_NAME,
                      cfg.MODEL_NAME, 'solver{}.prototxt'.format(postfix))
    check_if_exist('Solver', solver)

    # Set up caffe
    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)

    # Set imdb
    imdb_name = '{:s}_train'.format(cfg.DATASET_NAME)

    imdb = get_imdb(imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)

    # Start training
    train_net(solver, roidb, pretrained_model=pretrained_model,
              max_iters=max_iters)

def evaluate_faster_rcnn(conf_thresh, nms_thresh):
    """Evaluate a Faster R-CNN network on a image database."""
    # Set prototxt
    prototxt = osp.join(cfg.MODELS_DIR, cfg.DATASET_NAME, cfg.METHOD_NAME,
                        cfg.MODEL_NAME, 'test.prototxt')
    check_if_exist('Prototxt', prototxt)

    # Get most recent model
    test_model = get_model_path(cfg.OUTPUT_DIR, '.caffemodel', '_iter_')

    if test_model is None:
        print('No model found in `{:s}`.'.format(cfg.OUTPUT_DIR))
        sys.exit()

    caffe.set_mode_gpu()
    caffe.set_device(cfg.GPU_ID)
    net = caffe.Net(prototxt, caffe.TEST, weights=test_model)
    net.name = osp.splitext(osp.basename(test_model))[0]

    # Get imdb
    imdb_name = '{:s}_val'.format(cfg.DATASET_NAME)
    imdb = get_imdb(imdb_name)

    # results_dir = osp.join(cfg.OUTPUT_DIR, 'results')
    # imdb._do_pascal_voc_eval(results_dir)

    if not cfg.TEST.HAS_RPN:
        imdb.set_proposal_method(cfg.TEST.PROPOSAL_METHOD)

    test_net(net, imdb, conf_thresh, nms_thresh)



def detect_faster_rcnn(image_paths, result_file, conf_thresh,
                       nms_thresh, cpu_mode=False):
    """Detect object classes in given images with a Faster R-CNN network."""

    prototxt = osp.join(cfg.MODELS_DIR, cfg.DATASET_NAME, cfg.METHOD_NAME,
                        cfg.MODEL_NAME, 'test.prototxt')
    check_if_exist('Prototxt', prototxt)

    # Get model weights
    caffemodel = get_model_path(cfg.OUTPUT_DIR, '.caffemodel', '_iter_')

    if caffemodel is None:
        print('No model found in `{:s}`.'.format(cfg.OUTPUT_DIR))
        sys.exit()

    if cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(cfg.GPU_ID)

    # Load network
    net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)

    f = open(result_file, "w")
    _t = Timer()

    num_images = len(image_paths)
    num_classes = len(cfg.CLASSES)

    for i in range(0, num_images):
        # Load image
        path = image_paths[i]
        im = cv2.imread(path)
        image_name = path.split("/")[-1]

        # Crop borders for original baitcam images
        # if cfg.DATASET_NAME == 'baitcam':
        #     im = im[32:1504, 0:2043]

        # Detect all object classes and regress object bounds
        _t.tic()
        scores, boxes = im_detect(net, im)
        _t.toc()
        print('Detection took {:.3f}s for {:d} object proposals (image {:d}/{:d})'.format(
               _t.diff, boxes.shape[0], i+1, num_images))

        for cls_ind in range(1, num_classes): # skip background
            # Get results for class
            cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            detections = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)

            # Non maximum suppression to remove redundant overlapping detections
            keep = nms(detections, nms_thresh)
            detections = detections[keep, :]

            # Only keep detections with score higher than confidence threshold
            inds = np.where(detections[:, -1] >= conf_thresh)[0]

            # Write to results file
            for i in inds:
                bbox = detections[i, :4]
                score = float(detections[i, -1])
                xmin = int(np.around(bbox[0]))
                ymin = int(np.around(bbox[1]))
                xmax = int(np.around(bbox[2]))
                ymax = int(np.around(bbox[3]))

                # Compensate for cropped borders in original baitcam images
                # if cfg.DATASET_NAME == 'baitcam':
                #     ymin += 32
                #     ymax += 32

                # Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
                print('{:s} {:d} {:f} {:d} {:d} {:d} {:d}'.format(
                path, cls_ind, score, xmin, ymin, xmax, ymax),
                file=f)

    f.close()
