#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Show or save detection results output by detect.py.
@author: Håvard Thom
'''

from __future__ import print_function
import argparse
import numpy as np
import os.path as osp
import cv2

import matplotlib.pyplot as plt
from collections import OrderedDict
from google.protobuf import text_format
from caffe.proto import caffe_pb2 as cpb2

from utils import make_if_not_exist, check_if_exist

# Figure size for latex
def figsize(scale):
    fig_width_pt = 720
    inches_per_pt = 1.0/72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0)-1.0)/2.0
    fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
    fig_height = fig_width*golden_mean              # height in inches
    fig_size = [fig_width,fig_height]
    return fig_size

def visualize_results(img_results, classnames, vis_tresh, vis_class, skip):
    i = 0
    num_images = len(img_results)

    while 0 <= i < num_images:
        img_file = img_results.items()[i][0]
        results = img_results.items()[i][1]

        if not osp.exists(img_file):
            print('{} does not exist'.format(img_file))
            i+=1
            continue

        im = cv2.imread(img_file)
        im = im[:, :, (2, 1, 0)]

        fig = plt.figure(frameon=False)
        # fig.set_size_inches(15.97, 11.5)
        fig.figsize = figsize(1)

        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(im, aspect='auto')

        if classnames:
            # generate same number of colors as classes in classnames.
            num_classes = len(classnames)
        else:
            # generate 20 colors.
            num_classes = 20
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        dets = 0

        for res in results:
            if 'score' in res and vis_tresh and float(res["score"]) < vis_tresh:
                continue

            label = res['label']
            name = "class " + str(label)
            if classnames:
                name = classnames[label]

            if vis_class and name not in vis_class:
                continue

            color = colors[label % num_classes]
            bbox = res['bbox']
            ax.add_patch(
                        plt.Rectangle((bbox[0], bbox[1]), # x,y
                                      bbox[2] - bbox[0], # width
                                      bbox[3] - bbox[1], # height
                                      fill=False, edgecolor=color, linewidth=3.5)
                        )

            if 'score' in res:
                score = res['score']
                display_text = '{:s}: {:.0f}%'.format(name, np.round(score*100))
            else:
                display_text = name


            ax.text(bbox[0] + 5, bbox[1] - 15,
                    display_text,
                    bbox=dict(facecolor=color, alpha=0.5),
                    fontsize=15, color='black', family='Open Sans')

            dets += 1

        if skip and dets == 0:
            plt.close(fig)
            i+=1
            continue

        # Show or save image
        if not "out_file" in results[0]:
            print('\nShowing image {:d}/{:d} ({:s})'.format(i+1, num_images, img_file))
            print('Key click for next image or mouse click for previous image.')

            # Fullscreen
            # mng = plt.get_current_fig_manager()
            # mng.window.showMaximized()

            # Draw and wait for key or mouse press
            plt.draw()
            key = plt.waitforbuttonpress()
            plt.close(fig)

            if key == False:
                i-=1
                continue
        else:
            dest = results[0]["out_file"]
            print('Saving {:s} (image {:d}/{:d})'.format(dest, i+1, num_images))
            # fig.savefig(results[0]["out_file"], dpi=128)
            fig.savefig(dest)
            plt.close(fig)

        i+=1

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(
            description = "Show or save the detection results output by detect.py")
    parser.add_argument("result_file",
            help = "A file which contains all the detection results \
            (image path, class label, confidence score, xmin, ymin, xmax, ymax).")
    parser.add_argument("--labelmap", default="",
            help = "LabelMap file which contains classnames.", type=str)
    parser.add_argument("--vis_thresh", default=0.01, type=float,
            help = "If provided, only show/save detections with score higher than the threshold.")
    parser.add_argument("--vis_class", default=None,
            help = "If provided, only show/save specified class. Separate by ','")
    parser.add_argument('--save', dest='save',
            help='Save the images with detections in a results directory.',
            action='store_true', default=False)
    parser.add_argument('--skip', dest='skip',
            help='Skip images with no detections.',
            action='store_true', default=False)

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = parse_args()

    print('Called with args:')
    print(args)

    result_file = args.result_file
    labelmap_file = args.labelmap
    vis_thresh = args.vis_thresh
    vis_class = args.vis_class
    save = args.save
    skip = args.skip

    classnames = []
    if labelmap_file:
        check_if_exist('Label map file', labelmap_file)
        labelmap = cpb2.LabelMap()
        with open(labelmap_file,'r') as f:
            text_format.Merge(str(f.read()), labelmap)

        for item in labelmap.item:
            classname = str(item.display_name)
            classnames.append(classname)


    if save:
        save_dir = osp.splitext(result_file)[0]
        make_if_not_exist(save_dir)
        print('Saving to directory: {}'.format(save_dir))


    img_results = OrderedDict()
    with open(result_file, "r") as f:
        for line in f.readlines():
            img_path, label, score, xmin, ymin, xmax, ymax = line.strip("\n").split()

            result = dict()
            result["label"] = int(label)
            result["score"] = float(score)
            result["bbox"] = [float(xmin), float(ymin), float(xmax), float(ymax)]

            if save:
                out_file = osp.join(save_dir, '{}.png'.format(osp.splitext(osp.basename(img_path))[0]))
                result["out_file"] = out_file

            if img_path not in img_results:
                img_results[img_path] = [result]
            else:
                img_results[img_path].append(result)


    visualize_results(img_results, classnames, vis_thresh, vis_class, skip)
