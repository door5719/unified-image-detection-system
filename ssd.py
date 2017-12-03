#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Train, test and detect with a SSD network."""
from __future__ import print_function
import os.path as osp
import subprocess
import sys
import cv2
import numpy as np
import caffe
from caffe.model_libs import *

from fast_rcnn_utils.timer import Timer
from configs.config import cfg
from datasets.factory import get_imdb
from utils import make_if_not_exist, check_if_exist, get_model_path, visualize_filters

# Original function: https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_pascal.py
def AddExtraLayers(net, use_batchnorm=True, lr_mult=1):
    """Add extra layers on top of a "base" network (e.g. VGG or Inception)."""
    use_relu = True

    # Add additional convolutional layers.
    # 19 x 19 (300x300), 32 x 32 (512x512)
    from_layer = net.keys()[-1]

    # TODO(weiliu89): Construct name using the last layer to avoid duplication.
    # 10 x 10 (300x300), 16 x 16 (512x512), 19 x 19 (608x608)
    out_layer = 'conv6_1'
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu,
                256, 1, 0, 1, lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = 'conv6_2'
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu,
                512, 3, 1, 2, lr_mult=lr_mult)

    # 5 x 5 (300x300), 8 x 8 (512x512), 10 x 10 (608x608)
    from_layer = out_layer
    out_layer = 'conv7_1'
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu,
                128, 1, 0, 1, lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = 'conv7_2'
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu,
                256, 3, 1, 2, lr_mult=lr_mult)

    # 3 x 3 (300x300), 4 x 4 (512x512), 5 x 5 (608x608)
    from_layer = out_layer
    out_layer = 'conv8_1'
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu,
                128, 1, 0, 1, lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = 'conv8_2'
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu,
                256, 3, 0, 1, lr_mult=lr_mult)

    # 1 x 1 (300x300), 2 x 2 (512x512), 3 x 3 (608x608)
    from_layer = out_layer
    out_layer = 'conv9_1'
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu,
                128, 1, 0, 1, lr_mult=lr_mult)

    from_layer = out_layer
    out_layer = 'conv9_2'
    ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu,
                256, 3, 0, 1, lr_mult=lr_mult)

    if cfg.TRAIN.MAX_SIZE >= 512:
            # 1 x 1 (512x512), , 1 x 1 (608x608)
            from_layer = out_layer
            out_layer = 'conv10_1'
            ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu,
                        128, 1, 0, 1, lr_mult=lr_mult)

            from_layer = out_layer
            out_layer = 'conv10_2'
            ConvBNLayer(net, from_layer, out_layer, use_batchnorm, use_relu,
                        256, 4, 1, 1, lr_mult=lr_mult)

    return net

# Original function: https://github.com/weiliu89/caffe/blob/ssd/examples/ssd/ssd_pascal.py
# Modified by Håvard Thom
def create_ssd_model_definition(max_iters, conf_thresh, nms_thresh):
    """Create SSD network definition files based on config settings."""

    # Training and testing data created by data/data_utils/pascal_voc_to_ssd.py
    train_data = osp.join(cfg.DATA_DIR, cfg.DATASET_NAME, 'train_lmdb')
    test_data = osp.join(cfg.DATA_DIR, cfg.DATASET_NAME, 'val_lmdb')

    models_dir = osp.join(cfg.MODELS_DIR, cfg.DATASET_NAME,
                          cfg.METHOD_NAME, cfg.MODEL_NAME)

    make_if_not_exist(models_dir)
    check_if_exist('Training data', train_data)
    check_if_exist('Test data', test_data)

    # Directory which stores the detection results
    results_dir = osp.join(cfg.OUTPUT_DIR, 'results')

    # Model definition files.
    train_net_file = osp.join(models_dir, 'train.prototxt')
    test_net_file = osp.join(models_dir, 'test.prototxt')
    deploy_net_file = osp.join(models_dir, 'deploy.prototxt')
    train_solver_file = osp.join(models_dir, 'train_solver.prototxt')
    test_solver_file = osp.join(models_dir, 'test_solver.prototxt')

    # The name of the model
    model_name = '{}_ssd'.format(cfg.MODEL_NAME.lower())

    # Snapshot prefix.
    snapshot_prefix = osp.join(cfg.OUTPUT_DIR, model_name)

    # Stores the test image names and sizes
    name_size_file = osp.join(cfg.DATA_DIR, cfg.DATASET_NAME,
                              'ssd_ImageSets', 'val_name_size.txt')

    label_map_file = osp.join(cfg.DATA_DIR, cfg.DATASET_NAME,
                              '{}_labelmap.prototxt'.format(cfg.DATASET_NAME))

    # Specify the batch sampler.
    resize_width = cfg.TRAIN.MAX_SIZE
    resize_height = cfg.TRAIN.MAX_SIZE
    resize = '{}x{}'.format(resize_width, resize_height)
    batch_sampler = [
            {
                    'sampler': {
                            },
                    'max_trials': 1,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.1,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.3,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.5,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.7,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'min_jaccard_overlap': 0.9,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            {
                    'sampler': {
                            'min_scale': 0.3,
                            'max_scale': 1.0,
                            'min_aspect_ratio': 0.5,
                            'max_aspect_ratio': 2.0,
                            },
                    'sample_constraint': {
                            'max_jaccard_overlap': 1.0,
                            },
                    'max_trials': 50,
                    'max_sample': 1,
            },
            ]
    train_transform_param = {
            'mirror': True,
            # 'mean_value': [104, 117, 124],
            'mean_value': list(cfg.PIXEL_MEANS[0][0]),
            'force_color': True,
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': resize_height,
                    'width': resize_width,
                    # 'resize_mode': P.Resize.FIT_SMALL_SIZE,
                    # 'height': resize_height,
                    # 'width': resize_width,
                    # 'height_scale': resize_height,
                    # 'width_scale': resize_width,
                    'interp_mode': [
                            P.Resize.LINEAR,
                            P.Resize.AREA,
                            P.Resize.NEAREST,
                            P.Resize.CUBIC,
                            P.Resize.LANCZOS4,
                            ],
                    },
            'distort_param': {
                    'brightness_prob': 0.5,
                    'brightness_delta': 32,
                    'contrast_prob': 0.5,
                    'contrast_lower': 0.5,
                    'contrast_upper': 1.5,
                    'hue_prob': 0.5,
                    'hue_delta': 18,
                    'saturation_prob': 0.5,
                    'saturation_lower': 0.5,
                    'saturation_upper': 1.5,
                    'random_order_prob': 0.0,
                    },
            'expand_param': {
                    'prob': 0.5,
                    'max_expand_ratio': 4.0,
                    },
            'emit_constraint': {
                'emit_type': caffe_pb2.EmitConstraint.CENTER,
                }
            }
    test_transform_param = {
            # 'mean_value': [104, 117, 124],
            'mean_value': list(cfg.PIXEL_MEANS[0][0]),
            'force_color': True,
            'resize_param': {
                    'prob': 1,
                    'resize_mode': P.Resize.WARP,
                    'height': resize_height,
                    'width': resize_width,
                    # 'resize_mode': P.Resize.FIT_SMALL_SIZE,
                    # 'height': resize_height,
                    # 'width': resize_width,
                    # 'height_scale': resize_height,
                    # 'width_scale': resize_height,
                    'interp_mode': [P.Resize.LINEAR],
                    },
            }



    # If true, use batch norm for all newly added layers.
    # Currently only the non batch norm version has been tested.
    use_batchnorm = False
    lr_mult = 1
    # Use different initial learning rate.
    if use_batchnorm:
        base_lr = 0.0004
    else:
        # A learning rate for batch_size = 1, num_gpus = 1.
        base_lr = 0.00004

    # MultiBoxLoss parameters.
    num_classes = cfg.NUM_CLASSES
    share_location = True
    background_label_id = 0
    output_name_prefix = '{}_det_test_'.format(cfg.DATASET_NAME)
    train_on_diff_gt = False
    normalization_mode = P.Loss.VALID
    code_type = P.PriorBox.CENTER_SIZE
    ignore_cross_boundary_bbox = False
    mining_type = P.MultiBoxLoss.MAX_NEGATIVE
    neg_pos_ratio = 3.
    loc_weight = (neg_pos_ratio + 1.) / 4.
    multibox_loss_param = {
        'loc_loss_type': P.MultiBoxLoss.SMOOTH_L1,
        'conf_loss_type': P.MultiBoxLoss.SOFTMAX,
        'loc_weight': loc_weight,
        'num_classes': num_classes,
        'share_location': share_location,
        'match_type': P.MultiBoxLoss.PER_PREDICTION,
        'overlap_threshold': 0.5,
        'use_prior_for_matching': True,
        'background_label_id': background_label_id,
        'use_difficult_gt': train_on_diff_gt,
        'mining_type': mining_type,
        'neg_pos_ratio': neg_pos_ratio,
        'neg_overlap': 0.5,
        'code_type': code_type,
        'ignore_cross_boundary_bbox': ignore_cross_boundary_bbox,
        }
    loss_param = {
        'normalization': normalization_mode,
        }


    # parameters for generating priors.
    # minimum dimension of input image
    min_dim = cfg.TRAIN.MAX_SIZE
    # conv4_3 ==> 38 x 38 (300x300) ==> 64 x 64 (512x512)  ==> 76 x 76 (608x608)
    # fc7 ==> 19 x 19 (300x300) ==> 32 x 32 (512x512) ==> 38 x 38 (608x608)
    # conv6_2 ==> 10 x 10 (300x300) ==> 16 x 16 (512x512) ==> 19 x 19 (608x608)
    # conv7_2 ==> 5 x 5 (300x300) ==> 8 x 8 (512x512) ==> 10 x 10 (608x608)
    # conv8_2 ==> 3 x 3 (300x300) ==> 4 x 4 (512x512) ==> 5 x 5 (608x608)
    # conv9_2 ==> 1 x 1 (300x300) ==> 2 x 2 (512x512) ==> 3 x 3 (608x608)
    #                    conv10_2 ==> 1 x 1 (512x512) ==> 1 x 1 (608x608)


    if cfg.CUSTOM_ANCHORS:
        anchor_file = osp.join(cfg.DATA_DIR, cfg.DATASET_NAME, 'custom_anchor_boxes', '6_anchor_boxes.txt')
        if not osp.exists(anchor_file):
            print('Custom anchor boxes `{:s}` does not exist.'.format(anchor_file))
            print('Generate custom anchor boxes with '
                  'data/data_utils/k_means_anchor_boxes.py')
            sys.exit()

        # Read anchor file
        with open(anchor_file, 'r') as f:
            data = f.readlines()

        custom_anchors = []
        # aspect_ratio = []
        for i in range(1, len(data)):
            splt = data[i].split(',')
            anchor_width = float(splt[0])*min_dim
            anchor_height = float(splt[1])*min_dim
            # aspect_ratio.append(anchor_height/anchor_width)

            custom_anchors.append([anchor_width, anchor_height])


        custom_anchors = np.asarray(custom_anchors)
        print(custom_anchors)

        min_ratio = int(np.floor(np.min(custom_anchors)/min_dim*100))
        max_ratio = int(np.ceil(np.amax(custom_anchors)/min_dim*100))

        nb = 1
    else:
        # in percent %
        min_const = 10
        max_const = 20
        min_ratio = 20
        max_ratio = 90


        if min_dim == 512 or min_dim == 608:
            max_const = 10
            min_ratio = 10
            min_const = 4

        nb = 2

    mbox_source_layers = ['conv4_3', 'fc7', 'conv6_2', 'conv7_2',
                          'conv8_2', 'conv9_2']

    if min_dim == 512 or min_dim == 608:
        mbox_source_layers.append('conv10_2')

    step = int(np.floor((max_ratio - min_ratio) / (len(mbox_source_layers) - nb)))

    min_sizes = []
    max_sizes = []
    for ratio in xrange(min_ratio, max_ratio + 1, step):
        print(ratio)
        min_sizes.append(min_dim * ratio / 100.)
        max_sizes.append(min_dim * (ratio + step) / 100.)

    steps = [8, 16, 32, 64, 100, 300]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    # L2 normalize conv4_3.
    normalizations = [20, -1, -1, -1, -1, -1]

    if min_dim == 512:
        steps = [8, 16, 32, 64, 128, 256, 512]
        aspect_ratios.insert(2, [2, 3])
        normalizations.append(-1)
    elif min_dim == 608:
        steps = [8, 16, 32, 61, 122, 203, 608]
        aspect_ratios.insert(2, [2, 3])
        normalizations.append(-1)

    print("minsize: ", min_sizes)
    print("maxsize: ", max_sizes)
    if not cfg.CUSTOM_ANCHORS:
        min_sizes = [min_dim * min_const / 100.] + min_sizes
        max_sizes = [min_dim * max_const / 100.] + max_sizes
    print("minsize: ", min_sizes)
    print("maxsize: ", max_sizes)

    if min_dim != 300 and min_dim != 512:
        print('SSD anchor boxes are not optimized for size {}'.format(min_dim))

    # variance used to encode/decode prior bboxes.
    if code_type == P.PriorBox.CENTER_SIZE:
      prior_variance = [0.1, 0.1, 0.2, 0.2]
    else:
      prior_variance = [0.1]

    flip = True
    clip = False


    ### PRIOR CALCULATIONS THAT ARE DONE IN CAFFE LAYER
    for s in range(0, len(min_sizes)):
        min_size = min_sizes[s]
        # first prior: aspect_ratio = 1, size = min_size
        box_width = min_size
        box_height = min_size
        print('\nfirst: {} X {}'.format(box_width, box_height))

        if len(max_sizes) > 0:
            max_size = max_sizes[s]
            box_width = np.sqrt(min_size * max_size)
            box_height = np.sqrt(min_size * max_size)
            print('second: {} X {}'.format(box_width, box_height))

        for r in range(0, len(aspect_ratios[s])):
            ar = aspect_ratios[s][r]
            if np.fabs(ar - 1.) < 1e-6:
                continue

            box_width = min_size * np.sqrt(ar)
            box_height = min_size / np.sqrt(ar)
            print('rest: {} X {}'.format(box_width, box_height))


    # sys.exit()

    # Solver parameters.
    # Defining which GPUs to use.
    gpus = '{:d}'.format(cfg.GPU_ID)
    gpulist = gpus.split(',')
    num_gpus = len(gpulist)

    # Divide the mini-batch to different GPUs.
    batch_size = cfg.TRAIN.IMS_PER_BATCH
    accum_batch_size = cfg.TRAIN.BATCH_SIZE
    iter_size = accum_batch_size / batch_size
    solver_mode = P.Solver.CPU
    device_id = 0
    batch_size_per_device = batch_size
    if num_gpus > 0:
      batch_size_per_device = int(np.ceil(float(batch_size) / num_gpus))
      iter_size = int(np.ceil(float(accum_batch_size) /
                               (batch_size_per_device * num_gpus)))
      solver_mode = P.Solver.GPU
      device_id = int(gpulist[0])

    if normalization_mode == P.Loss.NONE:
      base_lr /= batch_size_per_device
    elif normalization_mode == P.Loss.VALID:
      base_lr *= 25. / loc_weight
    elif normalization_mode == P.Loss.FULL:
      # Roughly there are 2000 prior bboxes per image.
      # TODO(weiliu89): Estimate the exact # of priors.
      base_lr *= 2000.


    # Get number of test images from name_size_file
    num_test_image = sum(1 for line in open(name_size_file))
    test_batch_size = 8

    # Ideally test_batch_size should be divisible by num_test_image,
    test_iter = int(np.ceil(float(num_test_image) / test_batch_size))

    stepvalue = []
    stepvalue.append(int(np.ceil(max_iters*0.6667)))
    stepvalue.append(int(np.ceil(max_iters*0.8333)))
    stepvalue.append(max_iters)

    train_solver_param = {
        # Train parameters
        'base_lr': base_lr,
        'weight_decay': 0.0005,
        'lr_policy': 'multistep',
        'stepvalue': stepvalue,
        'gamma': 0.1,
        'momentum': 0.9,
        'iter_size': iter_size,
        'max_iter': max_iters,
        'snapshot': cfg.TRAIN.SNAPSHOT_ITERS,
        'display': 20,
        'average_loss': 10,
        'type': 'SGD',
        'solver_mode': solver_mode,
        'device_id': device_id,
        'debug_info': False,
        'snapshot_after_train': True,
        }

    test_solver_param = {
        # Test parameters
        'snapshot': 1,
        'snapshot_after_train': False,
        'test_iter': [test_iter],
        'test_interval': 1,
        'eval_type': 'detection',
        'ap_version': 'MaxIntegral',
        'test_initialization': True,
        }

    # Parameters for generating detection output.
    det_out_param = {
        'num_classes': num_classes,
        'share_location': share_location,
        'background_label_id': background_label_id,
        'nms_param': {'nms_threshold':nms_thresh, 'top_k': 200},
        'save_output_param': {
            'output_directory': results_dir,
            'output_name_prefix': output_name_prefix,
            'output_format': 'VOC',
            'label_map_file': label_map_file,
            'name_size_file': name_size_file,
            'num_test_image': num_test_image,
            },
        'keep_top_k': 50,
        'confidence_threshold': conf_thresh,
        'code_type': code_type,
        }

    # Parameters for evaluating detection results.
    det_eval_param = {
        'num_classes': num_classes,
        'background_label_id': background_label_id,
        'overlap_threshold': 0.5,
        'evaluate_difficult_gt': False,
        'name_size_file': name_size_file,
        }

    # Create train net.
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(train_data, batch_size=batch_size_per_device,
            train=True, output_label=True, label_map_file=label_map_file,
            transform_param=train_transform_param, batch_sampler=batch_sampler)

    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False)

    AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult)

    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
            use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
            aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
            num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)

    # Create the MultiBoxLossLayer.
    name = "mbox_loss"
    mbox_layers.append(net.label)
    net[name] = L.MultiBoxLoss(*mbox_layers, multibox_loss_param=multibox_loss_param,
            loss_param=loss_param, include=dict(phase=caffe_pb2.Phase.Value('TRAIN')),
            propagate_down=[True, True, False, False])

    with open(train_net_file, 'w') as f:
        print('name: "{}_train"'.format(model_name), file=f)
        print(net.to_proto(), file=f)

    # Create test net.
    net = caffe.NetSpec()
    net.data, net.label = CreateAnnotatedDataLayer(test_data, batch_size=test_batch_size,
            train=False, output_label=True, label_map_file=label_map_file,
            transform_param=test_transform_param)

    VGGNetBody(net, from_layer='data', fully_conv=True, reduced=True, dilated=True,
        dropout=False)

    AddExtraLayers(net, use_batchnorm, lr_mult=lr_mult)

    mbox_layers = CreateMultiBoxHead(net, data_layer='data', from_layers=mbox_source_layers,
            use_batchnorm=use_batchnorm, min_sizes=min_sizes, max_sizes=max_sizes,
            aspect_ratios=aspect_ratios, steps=steps, normalizations=normalizations,
            num_classes=num_classes, share_location=share_location, flip=flip, clip=clip,
            prior_variance=prior_variance, kernel_size=3, pad=1, lr_mult=lr_mult)

    conf_name = 'mbox_conf'
    if multibox_loss_param['conf_loss_type'] == P.MultiBoxLoss.SOFTMAX:
      reshape_name = '{}_reshape'.format(conf_name)
      net[reshape_name] = L.Reshape(net[conf_name], shape=dict(dim=[0, -1, num_classes]))
      softmax_name = '{}_softmax'.format(conf_name)
      net[softmax_name] = L.Softmax(net[reshape_name], axis=2)
      flatten_name = '{}_flatten'.format(conf_name)
      net[flatten_name] = L.Flatten(net[softmax_name], axis=1)
      mbox_layers[1] = net[flatten_name]
    elif multibox_loss_param['conf_loss_type'] == P.MultiBoxLoss.LOGISTIC:
      sigmoid_name = '{}_sigmoid'.format(conf_name)
      net[sigmoid_name] = L.Sigmoid(net[conf_name])
      mbox_layers[1] = net[sigmoid_name]

    net.detection_out = L.DetectionOutput(*mbox_layers,
        detection_output_param=det_out_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))
    net.detection_eval = L.DetectionEvaluate(net.detection_out, net.label,
        detection_evaluate_param=det_eval_param,
        include=dict(phase=caffe_pb2.Phase.Value('TEST')))

    with open(test_net_file, 'w') as f:
        print('name: "{}_test"'.format(model_name), file=f)
        print(net.to_proto(), file=f)

    # Create deploy net.
    # Remove the first and last layer from test net.
    deploy_net = net
    with open(deploy_net_file, 'w') as f:
        net_param = deploy_net.to_proto()
        # Remove the first (AnnotatedData) and last (DetectionEvaluate) layer from test net.
        del net_param.layer[0]
        del net_param.layer[-1]
        net_param.name = '{}_deploy'.format(model_name)
        net_param.input.extend(['data'])
        net_param.input_shape.extend([
            caffe_pb2.BlobShape(dim=[1, 3, resize_height, resize_width])])
        print(net_param, file=f)

    # Create training solver.
    train_solver = caffe_pb2.SolverParameter(
            train_net=train_net_file,
            snapshot_prefix=snapshot_prefix,
            **train_solver_param)

    with open(train_solver_file, 'w') as f:
        print(train_solver, file=f)

    # Create testing solver.
    test_solver = caffe_pb2.SolverParameter(
            train_net=train_net_file,
            test_net=[test_net_file],
            snapshot_prefix=snapshot_prefix,
            **test_solver_param)

    with open(test_solver_file, 'w') as f:
        print(test_solver, file=f)


def train_ssd(no_pretrained, resume_training=True):
    """Train a SSD network."""

    train_param = ''

    # Set pretrained model
    if not no_pretrained:
        pretrained_model = osp.join(cfg.DATA_DIR, 'imagenet_models',
                           '{:s}.caffemodel'.format(cfg.MODEL_NAME))
        check_if_exist('Pretrained model', pretrained_model)
        train_param = '--weights="{:s}"'.format(pretrained_model)

    # Set solver
    train_solver_file = osp.join(cfg.MODELS_DIR, cfg.DATASET_NAME,
                                 cfg.METHOD_NAME, cfg.MODEL_NAME, 'train_solver.prototxt')
    check_if_exist('Solver', train_solver_file)

    # Find most snapshot
    snapshot_file = get_model_path(cfg.OUTPUT_DIR, '.solverstate', '_iter_')

    # Load from most recently saved snapshot, if it exist
    if resume_training and snapshot_file != None:
        train_param = '--snapshot="{:s}"'.format(snapshot_file)

    # Train model
    cmd = './frameworks/caffe-rcnn-ssd/build/tools/caffe train \
           --solver="{}" {} --gpu="{}"\
          '.format(train_solver_file, train_param, cfg.GPU_ID)

    # subprocess.call(cmd, shell=True)

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1, shell=True)

    # Log training
    try:
        with process.stdout, open(cfg.OUTPUT_DIR + '/logfile.txt', 'ab') as f:
            for line in iter(process.stdout.readline, b''):
                print(line, end='')
                f.write(line)
    except Exception as e:
        print(e)

    process.wait()


def evaluate_ssd():
    """Evaluate a SSD network."""

    # Set results directory and solver
    results_dir = osp.join(cfg.OUTPUT_DIR, 'results')
    test_solver_file = osp.join(cfg.MODELS_DIR, cfg.DATASET_NAME,
                                cfg.METHOD_NAME, cfg.MODEL_NAME, 'test_solver.prototxt')

    make_if_not_exist(results_dir)
    check_if_exist('Solver', test_solver_file)

    # Find most recent model
    test_model = get_model_path(cfg.OUTPUT_DIR, '.caffemodel', '_iter_')

    if test_model is None:
        print('No model found in `{:s}`.'.format(cfg.OUTPUT_DIR))
        sys.exit()

    # Test model
    cmd = './frameworks/caffe-rcnn-ssd/build/tools/caffe train \
           --solver="{}" --weights="{}" --gpu="{}"\
          '.format(test_solver_file, test_model, cfg.GPU_ID)

    subprocess.call(cmd, shell=True)

    # Set imdb and do evaluation
    imdb_name = '{:s}_val'.format(cfg.DATASET_NAME)
    imdb = get_imdb(imdb_name)
    imdb._do_pascal_voc_eval(results_dir)


def detect_ssd(image_paths, result_file, conf_thresh, cpu_mode=False):
    """Detect object classes in given images with a SSD network."""

    prototxt = osp.join(cfg.MODELS_DIR, cfg.DATASET_NAME,
                        cfg.METHOD_NAME, cfg.MODEL_NAME, 'deploy.prototxt')

    check_if_exist('Model file', prototxt)

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

    # input preprocessing: 'data' is the name of the input blob == net.inputs[0]
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # Change input order to caffe format
    transformer.set_transpose('data', (2, 0, 1))
    # Set pixel means
    transformer.set_mean('data', cfg.PIXEL_MEANS[0][0])

    # Detect with a batch size of 1
    image_resize = cfg.TRAIN.MAX_SIZE
    net.blobs['data'].reshape(1,3,image_resize,image_resize)

    _t = Timer()
    f = open(result_file, 'w')

    num_images = len(image_paths)
    for i in range(0, num_images):
        path = image_paths[i]
        im = cv2.imread(path)
        image_name = path.split("/")[-1]

        # Crop borders for original baitcam images
        # if cfg.DATASET_NAME == 'baitcam':
        #     im = im[32:1504, 0:2043]

        # Preprocess image
        transformed_image = transformer.preprocess('data', im)
        net.blobs['data'].data[...] = transformed_image

        # Forward pass.
        _t.tic()
        detections = net.forward()['detection_out']
        _t.toc()
        print('Detection took {:.3f}s for {:d} object proposals (image {:d}/{:d})'.format(_t.diff, detections.shape[2], i+1, num_images))

        #### Feature Map visualization for Thesis
        # i=0
        # for layer_name, param in net.params.iteritems():
        #     if i==23:
        #         break
        #     print(layer_name + '\t' + str(param[0].data.shape), str(param[1].data.shape))
        #     i+=1
        # # filters = net.blobs['conv5_3'].data[0]
        # filters = net.blobs['conv6_2'].data[0, 5:14]
        # visualize_filters(filters)
        # break

        # Only keep detections with score higher than confidence threshold
        inds = np.where(detections[0,0,:,2] >= conf_thresh)[0]

        # Write to results file
        for i in inds:
            label = int(detections[0,0,:,1][i])
            score = float(detections[0,0,:,2][i])
            xmin = int(np.around(detections[0,0,:,3][i] * im.shape[1]))
            ymin = int(np.around(detections[0,0,:,4][i] * im.shape[0]))
            xmax = int(np.around(detections[0,0,:,5][i] * im.shape[1]))
            ymax = int(np.around(detections[0,0,:,6][i] * im.shape[0]))

            # Compensate for cropped borders in original baitcam images
            # if cfg.DATASET_NAME == 'baitcam':
            #     ymin += 32
            #     ymax += 32

            # Detection format: [image_id, label, score, xmin, ymin, xmax, ymax].
            print('{:s} {:d} {:f} {:d} {:d} {:d} {:d}'.format(
            path, label, score, xmin, ymin, xmax, ymax
            ), file=f)

    f.close()
