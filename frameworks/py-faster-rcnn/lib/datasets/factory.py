#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# Modified by Håvard Thom
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from datasets.custom_dataset import custom_dataset
from configs.config import cfg
import numpy as np

def add_imdb(dataset_name, splits):
    # global __sets
    # Set up <dataset>_<split>
    for split in splits:
        name = '{}_{}'.format(dataset_name, split)
        __sets[name] = (lambda split=split: custom_dataset(split))

def get_imdb(name):
    """Get an imdb (image database) by name."""
    if not __sets.has_key(name):
        raise KeyError('Unknown dataset: {}'.format(name))
    return __sets[name]()

def list_imdbs():
    """List all registered imdbs."""
    return __sets.keys()
