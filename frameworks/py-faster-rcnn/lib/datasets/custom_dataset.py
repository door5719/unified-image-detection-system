#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Håvard Thom
# --------------------------------------------------------
from __future__ import print_function
import os
import os.path as osp
from datasets.imdb import imdb
import datasets.ds_utils as ds_utils
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import fast_rcnn_utils.cython_bbox
import cPickle
import matplotlib.pyplot as plt
from itertools import cycle
import glob

from custom_dataset_eval import custom_dataset_eval
from configs.config import cfg

from utils import make_if_not_exist, check_if_exist

class custom_dataset(imdb):
    def __init__(self, image_set, dataset_path=None):
        imdb.__init__(self, image_set)
        self._image_set = image_set
        if dataset_path is None:
            self._dataset_path = osp.join(cfg.DATA_DIR, cfg.DATASET_NAME)
        else:
            self._dataset_path = dataset_path

        self._classes = cfg.CLASSES

        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))

        # Assume all images have same extension
        # self._image_ext = '.JPG'
        self._image_ext = osp.splitext(glob.glob(osp.join(self._dataset_path,
                                                          'images', '*'))[0])[1]

        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = self.selective_search_roidb

        # specific config options
        self.config = {'cleanup'     : False,
                       'use_diff'    : False,
                       'rpn_file'    : None,
                       'min_size'    : 2}


        check_if_exist('Dataset path', self._dataset_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        image_path = osp.join(self._dataset_path, 'images',
                                  index + self._image_ext)
        check_if_exist('Path', image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """
        # Example path to image set file:
        # self._dataset_path + pascal_ImageSets/val.txt
        image_set_file = osp.join(self._dataset_path, 'pascal_ImageSets',
                                      self._image_set + '.txt')
        check_if_exist('Path', image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        gt_roidb = [self._load_pascal_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def selective_search_roidb(self):
        """
        Return the database of selective search regions of interest.
        Ground-truth ROIs are also included.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = osp.join(self.cache_path,
                                  self.name + '_selective_search_roidb.pkl')

        if osp.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print('{} ss roidb loaded from {}'.format(self.name, cache_file))
            return roidb

        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            ss_roidb = self._load_selective_search_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, ss_roidb)
        else:
            roidb = self._load_selective_search_roidb(None)
        with open(cache_file, 'wb') as fid:
            cPickle.dump(roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print('wrote ss roidb to {}'.format(cache_file))

        return roidb

    def rpn_roidb(self):
        if self._image_set != 'test':
            gt_roidb = self.gt_roidb()
            rpn_roidb = self._load_rpn_roidb(gt_roidb)
            roidb = imdb.merge_roidbs(gt_roidb, rpn_roidb)
        else:
            roidb = self._load_rpn_roidb(None)

        return roidb

    def _load_rpn_roidb(self, gt_roidb):
        filename = self.config['rpn_file']
        print('loading {}'.format(filename))
        check_if_exist('rpn data', filename)
        with open(filename, 'rb') as f:
            box_list = cPickle.load(f)
        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_selective_search_roidb(self, gt_roidb):
        filename = osp.abspath(osp.join(cfg.DATA_DIR,
                                                'selective_search_data',
                                                self.name + '.mat'))
        check_if_exist('Selective search data', filename)
        raw_data = sio.loadmat(filename)['boxes'].ravel()

        box_list = []
        for i in xrange(raw_data.shape[0]):
            boxes = raw_data[i][:, (1, 0, 3, 2)] - 1
            keep = ds_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = ds_utils.filter_small_boxes(boxes, self.config['min_size'])
            boxes = boxes[keep, :]
            box_list.append(boxes)

        return self.create_roidb_from_box_list(box_list, gt_roidb)

    def _load_pascal_annotation(self, index):
        """
        Load image and bounding boxes info from XML file in the PASCAL VOC
        format.
        """
        filename = osp.join(self._dataset_path, 'pascal_Annotations',
                                index + '.xml')
        tree = ET.parse(filename)
        objs = tree.findall('object')
        if not self.config['use_diff']:
            # Exclude the samples labeled as difficult
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]
            objs = non_diff_objs
        num_objs = len(objs)

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)
        # "Seg" area for pascal is just the box area
        seg_areas = np.zeros((num_objs), dtype=np.float32)

        # Load object bounding boxes into a data frame.
        for ix, obj in enumerate(objs):
            bbox = obj.find('bndbox')
            # Make pixel indexes 0-based
            x1 = float(bbox.find('xmin').text) - 1
            y1 = float(bbox.find('ymin').text) - 1
            x2 = float(bbox.find('xmax').text) - 1
            y2 = float(bbox.find('ymax').text) - 1
            classname = self._class_to_ind[obj.find('name').text.strip()]
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = classname
            overlaps[ix, classname] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = scipy.sparse.csr_matrix(overlaps)

        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False,
                'seg_areas' : seg_areas}

    def _write_pascal_voc_results_files(self, all_boxes, results_dir):
        check_if_exist('Result directory', results_dir)
        # result structure: image_id score xmin ymin xmax ymax
        print('Writing results in PASCAL VOC format to {:s}'.format(results_dir))
        for cls_ind, classname in enumerate(self.classes):
            if classname == 'background':
                continue
            print('Writing {:s} {:s} results file'.format(classname, cfg.DATASET_NAME))
            filename = osp.join(results_dir,
                    '{:s}_det_test_{:s}.txt'.format(cfg.DATASET_NAME, classname))

            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(self.image_index):
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in xrange(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index, dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))


    def _do_pascal_voc_eval(self, results_dir):
        check_if_exist('Result directory', results_dir)

        annopath = osp.join(self._dataset_path, 'pascal_Annotations',
                                '{:s}.xml')

        imagesetfile = osp.join(self._dataset_path, 'pascal_ImageSets',
                                    self._image_set + '.txt')
                                    
        cachedir = osp.join(self._dataset_path, 'annotations_cache')
        aps = []

        fig = plt.figure()
        fig.set_size_inches(18.75, 10.25)
        colors = plt.cm.hsv(np.linspace(0, 1, self.num_classes)).tolist()

        f = open(osp.join(results_dir, 'mAP_results.txt'), 'w')

        for i in xrange(self.num_classes):
            classname = self._classes[i]

            if classname == 'background':
                continue
            filename = osp.join(results_dir,
                '{:s}_det_test_{:s}.txt'.format(cfg.DATASET_NAME, classname))
            rec, prec, ap = custom_dataset_eval(filename, annopath, imagesetfile,
                                                classname, cachedir, ovthresh=0.5)
            aps += [ap]

            print('Average Precision for {} = {:.4f}'.format(classname, ap))
            print('Average Precision for {} = {:.4f}'.format(classname, ap), file=f)

            color = colors[i % self.num_classes]

            plt.plot(rec, prec, color=color, lw=1.5,
                     label='{0:s} (AP = {1:0.2f})'.format(classname, ap))

        mAP = np.mean(aps)
        print('Mean Average Precision = {:.4f}'.format(mAP))
        print('Mean Average Precision = {:.4f}'.format(mAP), file=f)

        f.close()

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall curve')
        plt.legend(loc="lower left")
        print('Saving Precision-Recall curve to {:s}'.format(results_dir))
        fig.savefig(osp.join(results_dir, 'PRcurve.png'))

        new_output_dir = cfg.OUTPUT_DIR + '_{:.2f}'.format(mAP*100)
        os.rename(cfg.OUTPUT_DIR, new_output_dir)
        cfg.OUTPUT_DIR = new_output_dir


    def evaluate_detections(self, all_boxes):

        results_dir = cfg.OUTPUT_DIR + '/results'
        make_if_not_exist(results_dir)

        self._write_pascal_voc_results_files(all_boxes, results_dir)
        self._do_pascal_voc_eval(results_dir)

if __name__ == '__main__':
    from datasets.custom_dataset import custom_dataset
    d = custom_dataset('train')
    res = d.roidb
    from IPython import embed; embed()
