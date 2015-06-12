# -*- coding: utf-8 -*-
"""
SEGM.BOT: bag-of-things related functions.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

__version__ = 0.01
__author__ = 'Vlad Popovici'
__all__ = ['accumulate_bag']

import numpy as np

from sklearn.cluster import MiniBatchKMeans

from util.explore import random_window_on_regions, random_window, \
    sliding_window, sliding_window_on_regions
from util.misc import intg_image
from descriptors.txtgrey import HaarLikeDescriptor

def accumulate_bag(image, w_size, desc, max_bag_size, roi=None, strategy='random',
               it_start=(0,0), it_step=(1,1)):

    w_offset = (0, 0)
    if isinstance(desc, HaarLikeDescriptor):
        # this one works on integral images
        image = intg_image(image)
        # the sliding window should also be increased by 1:
        w_offset = (1, 1)
        w_size = (w_size[0] + w_offset[0], w_size[1] + w_offset[1])

    # create iterator:
    strategy = strategy.lower()
    if strategy == 'random':
        if roi is None:
            itw = random_window(image.shape, w_size, max_bag_size)
        else:
            itw = random_window_on_regions(image.shape, roi, w_size, max_bag_size)
    elif strategy == 'sliding':
        if roi is None:
            itw = sliding_window(image.shape, w_size, start=it_start, step=it_step)
        else:
            itw = sliding_window_on_regions(image.shape, roi, w_size, step=it_step)
    else:
        raise ValueError('Unknown strategy.')

    bag = []
    wnd = []
    for r in itw:
        # adjust if needed:
        r2 = (r[0], r[1] - w_offset[1], r[2], r[3] - w_offset[0])
        wnd.append(r2)
        bag.append(desc.compute(image[r[0]:r[1], r[2]:r[3]]))

    return bag, wnd


def build_codebook_kmeans(bag, codebook_size):
    vq = MiniBatchKMeans(n_clusters=code_book_size)  # vector quantizer
    X = np.array(bag[0])  # put all feature vectors in an array