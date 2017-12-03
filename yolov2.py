#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Train, test and detect with a YOLOv2 network."""
from __future__ import print_function
import numpy as np
import sys
import os
import os.path as osp
import subprocess
from datasets.factory import get_imdb

from configs.config import cfg
from utils import make_if_not_exist, check_if_exist, get_model_path

def create_yolov2_names_data_config():
    train_set = osp.join(cfg.DATA_DIR, cfg.DATASET_NAME,
                         'yolov2_ImageSets', 'train.txt')
    val_set = osp.join(cfg.DATA_DIR, cfg.DATASET_NAME,
                        'yolov2_ImageSets', 'val.txt')

    check_if_exist('YOLOv2 train set', train_set)
    check_if_exist('YOLOv2 validation set', val_set)

    results_dir = osp.join(cfg.OUTPUT_DIR, 'results')

    data_cfg = osp.join(cfg.OUTPUT_DIR, '{}.data'.format(cfg.DATASET_NAME))
    names_cfg = osp.join(cfg.OUTPUT_DIR, '{}.names'.format(cfg.DATASET_NAME))

    num_classes = cfg.NUM_CLASSES - 1 # No background class

    # Create names file for yolov2
    with open(names_cfg,'w') as f:
        for classname in cfg.CLASSES[1:]: # No background class
            print(classname, file=f)

    # Create data configuration file for yolov2
    with open(data_cfg,'w') as f:
        print('classes = {}'.format(num_classes), file=f)
        print('train = {}'.format(train_set), file=f)
        print('valid = {}'.format(val_set), file=f)
        # print('test = {}'.format(test_set), file=f)
        print('names = {}'.format(names_cfg), file=f)
        print('backup = {}'.format(cfg.OUTPUT_DIR), file=f)
        print('results = {}'.format(results_dir), file=f)
        print('eval = voc', file=f)

def create_yolov2_model_definition(max_iters):
    """Create YOLOv2 model definition and config files."""

    default_model_cfg = osp.join(cfg.MODELS_DIR, cfg.DATASET_NAME, cfg.METHOD_NAME,
                                 cfg.MODEL_NAME, '{}.cfg'.format(cfg.MODEL_NAME))

    check_if_exist('YOLOv2 default model config', default_model_cfg)

    # Create model config in output dir
    model_cfg = osp.join(cfg.OUTPUT_DIR, '{}.cfg'.format(cfg.DATASET_NAME))

    num_classes = cfg.NUM_CLASSES - 1 # No background class

    # Get custom anchors
    if cfg.CUSTOM_ANCHORS:
        anchor_file = osp.join(cfg.DATA_DIR, cfg.DATASET_NAME,
                               'custom_anchor_boxes', '5_anchor_boxes.txt')
        if not osp.exists(anchor_file):
            print('Custom anchor boxes `{:s}` does not exist.'.format(anchor_file))
            print('Generate custom anchor boxes with '
                  'data/data_utils/k_means_anchor_boxes.py')
            sys.exit()

        # Copy them to output dir
        copy(anchor_file, osp.join(cfg.OUTPUT_DIR, '5_anchor_boxes.txt'))

        # Read anchor file
        with open(anchor_file, 'r') as f:
            data = f.readlines()

        # Get custom anchors
        custom_anchors = ''
        for i in range(1, len(data)):
            splt = data[i].split(',')
            anchor_width = (float(splt[0])*cfg.TRAIN.MAX_SIZE)/32
            anchor_height = (float(splt[1])*cfg.TRAIN.SCALES[0])/32

            custom_anchors += '{:.6f}, {:.6f}, '.format(anchor_width, anchor_height)

        # Remove last comma
        custom_anchors = custom_anchors[:-2]


    # Get default model settings
    with open(default_model_cfg, 'r') as f:
        data = f.readlines()

    # Change model settings according to our dataset and config
    for i in range(len(data)):
        if 'batch' in data[i] and 'subdivisions' in data[i+1]:
            data[i] = 'batch={:d}\n'.format(cfg.TRAIN.IMS_PER_BATCH)
            data[i+1] = 'subdivisions={:d}\n'.format(cfg.TRAIN.BATCH_SIZE)
            data[i+2] = 'height={:d}\n'.format(cfg.TRAIN.SCALES[0])
            data[i+3] = 'width={:d}\n'.format(cfg.TRAIN.MAX_SIZE)
        elif 'max_batches' in data[i]:
            data[i] = 'max_batches={:d}\n'.format(max_iters)
            step1 = int(np.ceil(0.5*max_iters))
            step2 = int(np.ceil(0.75*max_iters))
            data[i+2] = 'steps={:d},{:d}\n'.format(step1, step2)
        elif 'filters' in data[i]:
            last_filters_idx = i
        elif 'anchors' in data[i]:
            if cfg.CUSTOM_ANCHORS:
               data[i] = 'anchors={:s}\n'.format(custom_anchors)
            data[i+2] = 'classes={:d}\n'.format(num_classes)
            num_anchors = len(data[i].split(','))/2
            data[i+4] = 'num={:d}\n'.format(num_anchors)
        elif 'random' in data[i]:
            data[i] = 'random=0\n'

    # last filter size is (num_classes + num_coords + 1)*num_anchors)
    last_filter_size = (num_classes + 5) * num_anchors
    data[last_filters_idx] = 'filters={:d}\n'.format(last_filter_size)

    # Write to our own model config
    with open(model_cfg, 'w') as f:
        f.writelines(data)


def train_yolov2(no_pretrained=False, resume_training=True):
    """Train a YOLOv2 network."""

    data_cfg = osp.join(cfg.OUTPUT_DIR, '{}.data'.format(cfg.DATASET_NAME))
    model_cfg = osp.join(cfg.OUTPUT_DIR, '{}.cfg'.format(cfg.DATASET_NAME))

    check_if_exist('YOLOv2 data config', data_cfg)
    check_if_exist('YOLOv2 model config', model_cfg)

    # Set pretrained model
    if no_pretrained:
        pretrained_model = None
    else:
        pretrained_model = osp.join(cfg.DATA_DIR, 'imagenet_models',
                                    '{:s}.weights'.format(cfg.MODEL_NAME))
        check_if_exist('Pretrained model', pretrained_model)

    # Find most recent snapshot
    snapshot_file = get_model_path(cfg.OUTPUT_DIR, '.weights', '_batch_')

    # Load from most recently saved snapshot, if it exist
    if resume_training and snapshot_file != None:
        pretrained_model = snapshot_file

    snapshot_prefix = cfg.MODEL_NAME + '_' + cfg.METHOD_NAME

    # Train model
    cmd = ('./frameworks/darknet/darknet-cpp detector train {:s} {:s} {} '
           '-gpus {:d} -out {:s}').format(data_cfg, model_cfg, pretrained_model,
                                          cfg.GPU_ID, snapshot_prefix)
    # subprocess.call(cmd, shell=True)

    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, bufsize=1)

    # Log training
    try:
        with process.stdout, open(cfg.OUTPUT_DIR + '/logfile.txt', 'ab') as f:
            for line in iter(process.stdout.readline, b''):
                print(line, end='')
                f.write(line)
    except Exception as e:
        print(e)

    process.wait()

def evaluate_yolov2(conf_thresh, nms_thresh):
    """Evaluate a YOLOv2 network."""

    results_dir = osp.join(cfg.OUTPUT_DIR, 'results')
    data_cfg = osp.join(cfg.OUTPUT_DIR, '{}.data'.format(cfg.DATASET_NAME))
    model_cfg = osp.join(cfg.OUTPUT_DIR, '{}.cfg'.format(cfg.DATASET_NAME))

    make_if_not_exist(results_dir)
    check_if_exist('YOLOv2 data config', data_cfg)
    check_if_exist('YOLOv2 model config', model_cfg)

    # Change model config for testing
    with open(model_cfg, 'r') as f:
        data = f.readlines()

    for i in range(len(data)):
        if 'height' in data[i]:
            data[i] = 'height={:d}\n'.format(cfg.TEST.SCALES[0])
            data[i+1] = 'width={:d}\n'.format(cfg.TEST.MAX_SIZE)

    with open(model_cfg, 'w') as f:
        f.writelines(data)

    # Find most recent model
    test_model = get_model_path(cfg.OUTPUT_DIR, '.weights', '_batch_')

    if test_model is None:
        print('No model found in `{:s}`.'.format(cfg.OUTPUT_DIR))
        sys.exit()

    result_file_prefix = '{}_det_test_'.format(cfg.DATASET_NAME)

    # Test model
    cmd = ('./frameworks/darknet/darknet-cpp detector valid {} {} {} -out {} '
           '-gpus {} -nms_thresh {:f}').format(data_cfg, model_cfg, test_model,
           result_file_prefix, cfg.GPU_ID, nms_thresh)

    subprocess.call(cmd, shell=True)

    # Set imdb and evaluate
    imdb_name = '{:s}_val'.format(cfg.DATASET_NAME)
    imdb = get_imdb(imdb_name)
    imdb._do_pascal_voc_eval(results_dir)

def detect_yolov2(image_paths, result_file, conf_thresh, nms_thresh):
    """Detect object classes in given images with a YOLOv2 network."""

    data_cfg = osp.join(cfg.OUTPUT_DIR, '{}.data'.format(cfg.DATASET_NAME))
    model_cfg = osp.join(cfg.OUTPUT_DIR, '{}.cfg'.format(cfg.DATASET_NAME))

    check_if_exist('YOLOv2 data config', data_cfg)
    check_if_exist('YOLOv2 model config', model_cfg)

    # Change model config for detection
    with open(model_cfg, 'r') as f:
        data = f.readlines()

    for i in range(len(data)):
        if 'height' in data[i]:
            data[i] = 'height={:d}\n'.format(cfg.TEST.SCALES[0])
            data[i+1] = 'width={:d}\n'.format(cfg.TEST.MAX_SIZE)

    with open(model_cfg, 'w') as f:
        f.writelines(data)

    # Get model weights
    model_weights = get_model_path(cfg.OUTPUT_DIR, '.weights', '_batch_')

    if model_weights is None:
        print('No model weights found in `{:s}`.'.format(cfg.OUTPUT_DIR))
        sys.exit()

    # Create temporary list file with image paths
    detect_list_file = osp.join(os.getcwd(), 'detect_files.txt')
    with open(detect_list_file, "w") as f:
        for path in image_paths:
            print(path, file=f)

    # Add detection list file to data config
    with open(data_cfg, "a") as f:
        print('detect = {:s}'.format(detect_list_file), file=f)

    cmd = ('./frameworks/darknet/darknet-cpp detector detect {} {} {} -out {} '
           '-thresh {} -nms_thresh {} -gpus {}').format(data_cfg, model_cfg,
           model_weights, result_file, conf_thresh, nms_thresh, cfg.GPU_ID)

    subprocess.call(cmd, shell=True)

    # Remove temporary list file with image paths
    os.remove(detect_list_file)
