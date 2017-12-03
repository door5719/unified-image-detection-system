#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Håvard Thom
# --------------------------------------------------------

"""Set up paths for the system."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add caffe to PYTHONPATH
caffe_path = osp.join(this_dir, 'frameworks', 'caffe-rcnn-ssd', 'python')
add_path(caffe_path)

# Add py-faster-rcnn lib to PYTHONPATH
lib_path = osp.join(this_dir, 'frameworks', 'py-faster-rcnn', 'lib')
add_path(lib_path)

# Add Unified Detection System src dir to PYTHONPATH
add_path(this_dir)
