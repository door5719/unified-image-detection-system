#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""
Unified detection system for training, evaluating and detecting with detection \
method, model and dataset of choice.
"""
from __future__ import print_function
import _init_paths
import argparse
import os.path as osp
import yaml
import sys
import pprint
import glob

from caffe.proto import caffe_pb2 as cpb2
from google.protobuf import text_format
from fast_rcnn_utils.timer import Timer
from datasets.factory import add_imdb, get_imdb
from configs.config import cfg, cfg_from_file, cfg_from_list
from utils import make_if_not_exist, check_if_exist, sec_2_hour_min_sec
from utils import get_classnames_from_labelmap, procedure_complete_print

from faster_rcnn import train_faster_rcnn, evaluate_faster_rcnn, detect_faster_rcnn
from ssd import train_ssd, evaluate_ssd, detect_ssd
from ssd import create_ssd_model_definition
from yolov2 import train_yolov2, evaluate_yolov2, detect_yolov2
from yolov2 import create_yolov2_model_definition, create_yolov2_names_data_config

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
    description='Unified detection system for training, evaluating and \
    detecting with framework and model of choice',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--method',
            help='Name of detection method to use for training',
            default='faster_rcnn', type=str)
    parser.add_argument('--model',
            help='Name of model to use for training',
            default='VGG16', type=str)
    parser.add_argument('--dataset',
            help='Name of dataset to use for training',
            default='baitcam', type=str)
    parser.add_argument('--cfg',
            help='A config file to use for training', type=str)
    parser.add_argument('--gpu',
            help='GPU device id to use', default=0, type=int)
    parser.add_argument('--nopretrained',
            help='Train without pretrained imagenet model',
            action='store_true', default=False)
    parser.add_argument('--max_iters',
            help='Number of iterations to train',
            default=100000, type=int)
    parser.add_argument('--eval',
            help='Evaluate a model from output directory',
            action='store_true', default=False)
    # parser.add_argument('--eval_set',
    #         help='Which imageset to evaluate ',
    #         default='val', type=str)
    parser.add_argument('--detect',
            help='Detect on images using a model from output directory',
            action='store_true', default=False)
    parser.add_argument('--output_dir',
            help='A output directory which contains a model for evaluation or \
            detection (e.g. output/baitcam/ssd/VGG16_reduced)', type=str)
    parser.add_argument('--image_dir',
            help='A directory which contains images for detection', type=str)
    parser.add_argument("--conf_thresh",
            help = "Only get detections with confidence score higher than the threshold.",
            default=0.005, type=float)
    parser.add_argument("--nms_thresh",
            help = "Detections with IoU overlap higher than the threshold will \
            be suppressed by Non-Maximum Suppression.",
            default=0.45, type=float)

    args = parser.parse_args()

    if args.eval and args.output_dir is None:
        parser.error("--eval requires --output_dir.")
    if args.detect and args.output_dir is None:
        parser.error("--detect requires --output_dir.")
    if args.detect and args.image_dir is None:
        parser.error("--detect requires --image_dir.")

    return args


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    method_name = args.method
    model_name = args.model
    dataset_name = args.dataset
    cfg_file = args.cfg
    gpu_id = args.gpu
    no_pretrained = args.nopretrained
    max_iters = args.max_iters
    evaluate = args.eval
    # eval_set = args.eval_set
    detect = args.detect
    output_dir = args.output_dir
    image_dir = args.image_dir
    conf_thresh = args.conf_thresh
    nms_thresh = args.nms_thresh


    _t = Timer()

    if not evaluate and not detect:
        if cfg_file == None:
            cfg_file = osp.join('configs', method_name, 'default.yml')
            print('No config file given, '
                  'using default config: {:s}'.format(cfg_file))

        check_if_exist('Config', cfg_file)

        extra_cfg = ('METHOD_NAME {:s} MODEL_NAME {:s} '
                     'DATASET_NAME {:s} GPU_ID {:d}'.format(method_name,
                     model_name, dataset_name, gpu_id))

        set_cfgs = extra_cfg.split()

        # Update config
        cfg_from_file(cfg_file)
        cfg_from_list(set_cfgs)

        # Set and create output dir
        cfg.OUTPUT_DIR = osp.join(cfg.OUTPUT_DIR, cfg.DATASET_NAME, cfg.METHOD_NAME, cfg.MODEL_NAME)
        make_if_not_exist(cfg.OUTPUT_DIR)

        # Get classes from label map
        label_map_file = osp.join(cfg.DATA_DIR, cfg.DATASET_NAME,
                                  '{}_labelmap.prototxt'.format(cfg.DATASET_NAME))

        cfg.CLASSES = get_classnames_from_labelmap(label_map_file)
        cfg.NUM_CLASSES = len(cfg.CLASSES)

        # Dump full config to output dir
        dst = osp.join(cfg.OUTPUT_DIR, 'config.yml')
        with open(dst, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)
    else:
        # Get config from given output directory
        cfg_file = osp.join(output_dir, 'config.yml')

        check_if_exist('Config', cfg_file)

        extra_cfg = 'GPU_ID {:d}'.format(gpu_id)
        set_cfgs = extra_cfg.split()

        # Update config
        cfg_from_file(cfg_file)
        cfg_from_list(set_cfgs)

        cfg.OUTPUT_DIR = osp.abspath(output_dir)

    print('Using config:')
    pprint.pprint(cfg)

    # Add image database for evaluation (and training faster_rcnn)
    add_imdb(cfg.DATASET_NAME, ['train', 'val', 'test'])

    # Get image paths for detection
    if detect:
        image_dir = osp.abspath(image_dir)
        extensions = ['*.png', '*.jpg', '*.JPEG', '*.JPG']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(osp.join(image_dir, ext)))

        result_file = image_dir + '_detections.txt'


    # Set imdb and evaluate
    # results_dir = osp.join(cfg.OUTPUT_DIR, 'results')
    # imdb_name = '{:s}_val'.format(cfg.DATASET_NAME)
    # imdb = get_imdb(imdb_name)
    # imdb._do_pascal_voc_eval(results_dir)
    #
    # sys.exit()

    if cfg.METHOD_NAME == 'faster_rcnn':
        if cfg.MODEL_NAME != 'VGG16' and \
           cfg.MODEL_NAME != 'ResNet101_bn-scale-merged' and \
           cfg.MODEL_NAME != 'ResNet50' and \
           cfg.MODEL_NAME != 'VGG_CNN_M_1024' and \
           cfg.MODEL_NAME != 'ZF':
           print('Faster R-CNN detection method does not currently support model: {:s}'
                 ' (supported models: VGG16, ResNet101_bn-scale-merged, '
                 'ResNet50, VGG_CNN_M_1024 and ZF).'.format(cfg.MODEL_NAME))
           sys.exit()

        if not evaluate and not detect:
            _t.tic()
            train_faster_rcnn(no_pretrained, max_iters)
            _t.toc()
            procedure_complete_print('Training', _t.diff, cfg.OUTPUT_DIR, log=True)

        if not detect: # Evaluate directly after training
            _t.tic()
            evaluate_faster_rcnn(conf_thresh, nms_thresh)
            _t.toc()
            procedure_complete_print('Evaluation', _t.diff, cfg.OUTPUT_DIR)
        else:
            _t.tic()
            detect_faster_rcnn(image_paths, result_file, conf_thresh, nms_thresh)
            _t.toc()
            procedure_complete_print('Detection', _t.diff, result_file)

    elif cfg.METHOD_NAME == 'ssd':
        if cfg.MODEL_NAME != 'VGG16_reduced':
            print('SSD detection method does not currently support model: {:s}'
                  ' (supported models: VGG16_reduced).'.format(cfg.MODEL_NAME))
            sys.exit()

        create_ssd_model_definition(max_iters, conf_thresh, nms_thresh)

        if not evaluate and not detect:
            _t.tic()
            train_ssd(no_pretrained)
            _t.toc()
            procedure_complete_print('Training', _t.diff, cfg.OUTPUT_DIR, log=True)

        if not detect: # Evaluate directly after training
            _t.tic()
            evaluate_ssd()
            _t.toc()
            procedure_complete_print('Evaluation', _t.diff, cfg.OUTPUT_DIR)
        else:
            _t.tic()
            detect_ssd(image_paths, result_file, conf_thresh)
            _t.toc()
            procedure_complete_print('Detection', _t.diff, result_file)

    elif cfg.METHOD_NAME == 'yolov2':
        if cfg.MODEL_NAME != 'Darknet19':
            print('YOLOv2 detection method does not currently support model: '
                  '{:s} (supported models: Darknet19).'.format(cfg.MODEL_NAME))
            sys.exit()

        create_yolov2_names_data_config()

        if not evaluate and not detect:

            create_yolov2_model_definition(max_iters)

            _t.tic()
            train_yolov2(no_pretrained)
            _t.toc()
            procedure_complete_print('Training', _t.diff, cfg.OUTPUT_DIR, log=True)

        if not detect: # Evaluate directly after training
            _t.tic()
            evaluate_yolov2(conf_thresh, nms_thresh)
            _t.toc()
            procedure_complete_print('Evaluation', _t.diff, cfg.OUTPUT_DIR)
        else:
            _t.tic()
            detect_yolov2(image_paths, result_file, conf_thresh, nms_thresh)
            _t.toc()
            procedure_complete_print('Detection', _t.diff, result_file)

    else:
        print('Detection method {} not supported'.format(cfg.METHOD_NAME))
        sys.exit()
