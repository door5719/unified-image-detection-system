#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Utility functions for unified detection system."""
from __future__ import print_function
import os
import os.path as osp
import sys
import numpy as np
from pprint import pprint

import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2 as cpb2
from google.protobuf import text_format

def make_if_not_exist(path):
    """Create directory tree if it does not exist"""
    if not osp.exists(path):
        os.makedirs(path)

def check_if_exist(prefix, filename):
    """Check if a file exists"""
    if not osp.exists(filename):
        print('{:s} `{:s}` does not exist.'.format(prefix, filename))
        sys.exit()

def sec_2_hour_min_sec(seconds):
    """Convert seconds to hours, minutes and seconds"""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h,m,s

def get_model_path(model_dir, extension, infix):
    """Find most recent model in directory"""
    max_number = 0
    model_path = None
    for filename in os.listdir(model_dir):
      if filename.endswith(extension):
        basename = osp.splitext(filename)[0]
        number = int(basename.split(infix)[1])
        if number > max_number:
          max_number = number
          model_path = osp.join(model_dir, filename)
    return model_path

def get_classnames_from_labelmap(label_map_file):
    """Get classnames from labelmap"""
    check_if_exist('Label map file', label_map_file)

    labelmap = cpb2.LabelMap()
    with open(label_map_file,'r') as f:
        text_format.Merge(str(f.read()), labelmap)

    classnames = []
    for item in labelmap.item:
        classnames.append(str(item.display_name))

    if len(classnames) == 0:
        print('No classnames found in labelmap: {:s}.'.format(label_map_file))
        sys.exit()

    return classnames


# Original function: http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb
def visualize_filters(data):
    """Take an array of shape (n, height, width) or (n, height, width, 3)
       and visualize each (height, width) feature map in a grid of size
       approx. sqrt(n) by sqrt(n)"""

    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 1), (0, 1))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)

    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    print(data.shape)

    # fig=figure(facecolor='black')

    plt.axis('off')

    plt.imshow(data, cmap='jet')
    plt.show()

def procedure_complete_print(procedure_name, seconds, output_dir, log=False):
    h, m, s = sec_2_hour_min_sec(seconds)
    print('Total {:s} time: {:.0f}h {:.2f}m {:.2f}s'.format(
            procedure_name, h, m, s))

    print('{:s} complete, results are stored in {:s}'.format(
            procedure_name, output_dir))

    if log:
        with open(output_dir + '/logfile.txt', 'ab') as f:
            print('Total {:s} time: {:.0f}h {:.2f}m {:.2f}s'.format(
                  procedure_name, h, m, s), file=f)

            print('{:s} complete, results are stored in {:s}'.format(
                  procedure_name, output_dir), file=f)
