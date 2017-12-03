#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Written by Håvard Thom
"""Convert PASCAL VOC data to YOLOv2 data (create imagesets and annotations for YOLOv2)."""
import sys
import argparse
import random
import shutil
import os
import os.path as osp
import xml.etree.ElementTree as ET
import glob

src_dir = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(src_dir)

from utils import make_if_not_exist, check_if_exist, get_classnames_from_labelmap

def pascal_2_yolov2_bbox(image_size, pascal_bbox):
    """Convert PASCAL bounding box to YOLOv2 bounding box"""
    # xmin,ymin,xmax,ymax -> xcenter,ycenter,width,height
    x = (pascal_bbox[0] + pascal_bbox[2])/2.0
    y = (pascal_bbox[1] + pascal_bbox[3])/2.0
    w = pascal_bbox[2] - pascal_bbox[0]
    h = pascal_bbox[3] - pascal_bbox[1]

    # Normalize from original coordinates to [0,1] coordinates
    dw = 1./image_size[0]
    dh = 1./image_size[1]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def pascal_2_yolov2_annotation(pascal_annotation_path, yolov2_annotation_path, classnames):
    """Convert PASCAL annotation to YOLOv2 annotation"""

    pascal_annotation = open(pascal_annotation_path)
    yolov2_annotation = open(yolov2_annotation_path, 'w')

    tree = ET.parse(pascal_annotation)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    image_size = (w,h)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        classname = obj.find('name').text
        if classname not in classnames:
            print('{:s} contains invalid class: '
                  '{:s}'.format(pascal_annotation_path, classname))
            sys.exit()
        elif int(difficult) == 1:
            continue

        cls_id = classnames.index(classname)

        xmlbox = obj.find('bndbox')
        pascal_bbox = (float(xmlbox.find('xmin').text),
        float(xmlbox.find('ymin').text),
        float(xmlbox.find('xmax').text),
        float(xmlbox.find('ymax').text))

        yolov2_bbox = pascal_2_yolov2_bbox(image_size, pascal_bbox)

        yolov2_annotation.write(str(cls_id) + " " +
                                " ".join([str(a) for a in yolov2_bbox]) + '\n')

    yolov2_annotation.close()

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
    description='Convert PASCAL VOC data to YOLOv2 data \
    (create imagesets and annotations for YOLOv2)',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('dataset_dir',
                        help="The dataset directory which contains 'images', \
                        'pascal_ImageSets' and 'pascal_Annotations' folders. \
                        (Example: ~/datasets/baitcam_dataset)", type=str)
    parser.add_argument("label_map_file",
                        help = "Label map file which contains classnames.", type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    dataset_dir = args.dataset_dir
    label_map_file = args.label_map_file

    # Remove trailing slash
    dataset_dir = dataset_dir.rstrip('/')

    image_dir = osp.join(dataset_dir, 'images')
    pascal_annotations_dir = osp.join(dataset_dir, 'pascal_Annotations')
    pascal_imageset_dir = osp.join(dataset_dir, 'pascal_ImageSets')

    check_if_exist('Dataset directory', dataset_dir)
    check_if_exist('Image directory', image_dir)
    check_if_exist('Annotation directory', pascal_annotations_dir)
    check_if_exist('Imageset directory', pascal_imageset_dir)

    imagesets = [osp.basename(s) for s in glob.glob(pascal_imageset_dir + '/*.txt')]
    # imagesets = ['train', 'val']

    classnames = get_classnames_from_labelmap(label_map_file)
    # classnames = ['ArcticFox', 'Crow', 'Eagle', 'GoldenEagle', 'Raven',
    #                 'RedFox', 'Reindeer', 'SnowyOwl', 'Wolverine']

    # Remove background class
    if 'background' in classnames:
        classnames.remove('background')

    # Assume all images have same extension
    image_extension = osp.splitext(glob.glob(image_dir + '/*')[0])[1]

    yolov2_annotations_dir = osp.join(dataset_dir, 'yolov2_Annotations')
    yolov2_imageset_dir = osp.join(dataset_dir, 'yolov2_ImageSets')

    # # Remove existing labels
    # if osp.exists(yolov2_annotations_dir):
    #   shutil.rmtree(yolov2_annotations_dir)

    make_if_not_exist(yolov2_annotations_dir)
    make_if_not_exist(yolov2_imageset_dir)

    # Create imagesets for YOLOv2
    for imageset in imagesets:
        print('Creating YOLOv2 Imageset: {:s}'.format(imageset))
        pascal_image_ids = open(osp.join(pascal_imageset_dir,
                                imageset)).read().strip().split()

        yolov2_imageset_file = open(osp.join(yolov2_imageset_dir, imageset), 'w')

        # Shuffle train file
        if imageset == 'train.txt':
            random.shuffle(pascal_image_ids)

        # Convert PASCAL annotations to YOLOv2 annotations
        print('Converting PASCAL annotations to YOLOv2 annotations...')
        for image_id in pascal_image_ids:
            pascal_annotation_path = osp.join(pascal_annotations_dir,
                                              '{}.xml'.format(image_id))

            check_if_exist('Pascal annotation', pascal_annotation_path)

            yolov2_imageset_file.write(image_dir + '/' + image_id +
                                       image_extension + '\n')

            yolov2_annotation_path = osp.join(yolov2_annotations_dir,
                                              '{}.txt'.format(image_id))

            pascal_2_yolov2_annotation(pascal_annotation_path,
                                       yolov2_annotation_path, classnames)

        yolov2_imageset_file.close()


    # Create symbolic link from dataset directory to src/data directory
    link_dir = os.path.join(src_dir, 'data', os.path.basename(dataset_dir))
    if os.path.exists(link_dir):
      os.unlink(link_dir)

    print('Creating symbolic link from {:s} to {:s}'.format(dataset_dir, link_dir))
    os.symlink(dataset_dir, link_dir)
