#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Convert PASCAL VOC data to SSD data (create imagesets and LMDBs for SSD)."""
import sys
import argparse
import random
import shutil
import os
import os.path as osp
import subprocess
import glob
from caffe.proto import caffe_pb2
from google.protobuf import text_format

src_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(src_dir)

from utils import make_if_not_exist, check_if_exist, get_classnames_from_labelmap

def create_lmdb(dataset_dir, label_map_file, caffe_root, imagesets, resize_height, resize_width):
    anno_type = 'detection'
    label_type = 'xml'
    check_label = True
    min_dim = 0
    max_dim = 0
    resize_height = resize_height
    resize_width = resize_width
    backend = 'lmdb'
    shuffle = False
    check_size = False
    encode_type = 'jpg'
    encoded = True
    gray = False

    check_if_exist('Label map file', label_map_file)
    # Check if label map file has classes
    get_classnames_from_labelmap(label_map_file)

    # Create lmdb data for SSD
    for imageset in imagesets:
        print('Creating lmdb for Imageset: {:s}'.format(imageset))
        imageset_file = osp.join(ssd_imageset_dir, imageset)
        out_dir = osp.join(dataset_dir, '{}_{}'.format(
                           os.path.splitext(imageset)[0], backend))

        if osp.exists(out_dir):
          shutil.rmtree(out_dir)

        cmd = "{}/build/tools/convert_annoset" \
              " --anno_type={}" \
              " --label_type={}" \
              " --label_map_file={}" \
              " --check_label={}" \
              " --min_dim={}" \
              " --max_dim={}" \
              " --resize_height={}" \
              " --resize_width={}" \
              " --backend={}" \
              " --shuffle={}" \
              " --check_size={}" \
              " --encode_type={}" \
              " --encoded={}" \
              " --gray={}" \
              " {}/ {} {}" \
              .format(caffe_root, anno_type, label_type, label_map_file, check_label,
                  min_dim, max_dim, resize_height, resize_width, backend, shuffle,
                  check_size, encode_type, encoded, gray, dataset_dir, imageset_file, out_dir)

        print cmd
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        output = process.communicate()[0]

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
    description='Convert PASCAL VOC data to SSD data \
    (create imagesets and lmdb for SSD)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('dataset_dir',
                        help="The dataset directory which contains 'images', \
                        'pascal_ImageSets' and 'pascal_Annotations' folders. \
                        (Example: ~/datasets/baitcam_dataset)", type=str)
    parser.add_argument("label_map_file",
                        help = "Label map file which contains classnames.", type=str)
    parser.add_argument('--resize-height', default = 0, type = int,
                        help='Height images are resized to (in the lmdb).')
    parser.add_argument('--resize-width', default = 0, type = int,
                        help='Width images are resized to (in the lmdb).')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    # cmd = '#!/bin/bash exec &> >(tee -a "logfile.txt")'
    # subprocess.call(cmd, shell=True)

    print('Called with args:')
    print(args)

    dataset_dir = args.dataset_dir
    label_map_file = args.label_map_file
    resize_width = args.resize_width
    resize_height = args.resize_height

    # Remove trailing slash
    dataset_dir = dataset_dir.rstrip('/')

    label_map_file = osp.abspath(label_map_file)

    image_dir = osp.join(dataset_dir, 'images')
    pascal_annotations_dir = osp.join(dataset_dir, 'pascal_Annotations')
    pascal_imageset_dir = osp.join(dataset_dir, 'pascal_ImageSets')

    check_if_exist('Dataset directory', dataset_dir)
    check_if_exist('Image directory', image_dir)
    check_if_exist('Annotation directory', pascal_annotations_dir)
    check_if_exist('Imageset directory', pascal_imageset_dir)

    # src_dir = osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))
    caffe_root = osp.join(src_dir, 'frameworks', 'caffe-rcnn-ssd')

    imagesets = [os.path.basename(s) for s in glob.glob(pascal_imageset_dir + '/*.txt')]
    # imagesets = ['train', 'val']

    # Assume all images have same extension
    image_extension = osp.splitext(glob.glob(image_dir + '/*')[0])[1]

    ssd_imageset_dir = osp.join(dataset_dir, 'ssd_ImageSets')
    make_if_not_exist(ssd_imageset_dir)

    # Create imagesets for SSD
    for imageset in imagesets:
        print('Creating SSD Imageset: {:s}'.format(imageset))
        pascal_image_ids = open(osp.join(pascal_imageset_dir,
                                imageset)).read().strip().split()

        ssd_imageset_filename = osp.join(ssd_imageset_dir, imageset)
        ssd_imageset_file = open(ssd_imageset_filename, 'w')

        # Shuffle train file
        if imageset == 'train.txt':
            random.shuffle(pascal_image_ids)

        for image_id in pascal_image_ids:
            ssd_imageset_file.write('images/{}{} pascal_Annotations/{}.xml\n'.format(
                                    image_id, image_extension, image_id))

        ssd_imageset_file.close()

         # Generate image name and size infomation for validation set
        if imageset == 'val.txt':
            print('Creating val_name_size file for SSD...')
            cmd = "{}/build/tools/get_image_size {}/ {} {}/val_name_size.txt".format(
                    caffe_root, dataset_dir, ssd_imageset_filename, ssd_imageset_dir)

            print cmd
            process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
            output = process.communicate()[0]


    # Create lmdb
    create_lmdb(dataset_dir, label_map_file, caffe_root, imagesets,
                resize_height, resize_width)

    # Create symbolic link from dataset directory to src/data directory
    link_dir = os.path.join(src_dir, 'data', os.path.basename(dataset_dir))
    if os.path.exists(link_dir):
      os.unlink(link_dir)

    print('Creating symbolic link from {:s} to {:s}'.format(dataset_dir, link_dir))
    os.symlink(dataset_dir, link_dir)
