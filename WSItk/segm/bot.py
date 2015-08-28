# -*- coding: utf-8 -*-
"""
SEGM.BOT: bag-of-things related functions.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__version__ = 0.01
__author__ = 'Vlad Popovici'
__all__ = ['grow_bag_from_new_image', 'grow_bag_with_new_features',
           'grow_bag_from_images', 'read_bag', 'build_codebook_kmeans_online']

import numpy as np

from sklearn.cluster import MiniBatchKMeans
import skimage.io
from skimage.util import img_as_float

from util.explore import random_window_on_regions, random_window, \
    sliding_window, sliding_window_on_regions
from util.misc import intg_image
from descriptors.txtgrey import HaarLikeDescriptor
from ml.gap import gap

def grow_bag_from_new_image(image, desc, w_size, n_obj, **kwargs):
    """
    Extracts local descriptors from a new image.

    :param image: numpy.array
        Image data (single channel).
    :param desc: LocalDescriptor
        Local descriptor for feature extraction.
    :param w_size: tuple
        (width, height) of the sub-windows from the image.
    :param n_obj: int
        Maximum number of objects to be added to the bag.
    :param kwargs: dict
        Other parameters:
        'roi': region of interest (default: None)
        'sampling_strategy': how the image should be sampled:
            'random' for random sampling
            'sliding' for systematic, sliding window scanning
             of the image
        'it_start': where the scanning of the image starts (for
            sliding window sampling strategy) (default (0,0))
        'it_step': step from one window to the next (for
            sliding window sampling strategy) (default (1,1))
        'discard_empty': (boolean) whether an empy patch should still
            be processed or simply discarded. Default: False
    :return: dict
        A dictionary with two elements:
            <name of the descriptor>: list
            'regions': list
        The first list contains the feature descriptors.
        The second list contains the corresponding window positions.

    See also: grow_bag_with_new_features
    """

    if 'roi' not in kwargs:
        roi = None
    else:
        roi = kwargs['roi']

    if 'it_start' not in kwargs:
        it_start = (0,0)
    else:
        it_start = kwargs['it_start']

    if 'it_step' not in kwargs:
        it_step = (1,1)
    else:
        it_step = kwargs['it_step']

    if 'sampling_strategy' not in kwargs:
        sampling_strategy = 'random'
    else:
        sampling_strategy = kwargs['sampling_strategy']
        
    if 'discard_empty' in kwargs:
        discard_empty = kwargs['discard_empty']
    else:
        discard_empty = False

    w_offset = (0, 0)
    if isinstance(desc, HaarLikeDescriptor):
        # this one works on integral images
        image = intg_image(image)
        # the sliding window should also be increased by 1:
        w_offset = (1, 1)
        w_size = (w_size[0] + w_offset[0], w_size[1] + w_offset[1])

    # create iterator:
    sampling_strategy = sampling_strategy.lower()
    if sampling_strategy == 'random':
        if roi is None:
            itw = random_window(image.shape, w_size, n_obj)
        else:
            itw = random_window_on_regions(image.shape, roi, w_size, n_obj)
    elif sampling_strategy == 'sliding':
        if roi is None:
            itw = sliding_window(image.shape, w_size, start=it_start, step=it_step)
        else:
            itw = sliding_window_on_regions(image.shape, roi, w_size, step=it_step)
    else:
        raise ValueError('Unknown strategy.')

    bag = []
    wnd = []
    n = 0

    for r in itw:
        if discard_empty and image[r[0]:r[1], r[2]:r[3]].sum() < 1e-16:
            continue

        # adjust if needed:
        r2 = (r[0], r[1] - w_offset[1], r[2], r[3] - w_offset[0])
        wnd.append(r2)
        bag.append(desc.compute(image[r[0]:r[1], r[2]:r[3]]))

        n += 1
        if n > n_obj:
            break

    return {desc.name: bag, 'regs': wnd}


def grow_bag_with_new_features(image, regions, desc):
    """
    Returns the features corresponding to a list of regions. This
    is usually used for adding new features to an existing bag,
    where the list of regions has been obtained from an iterator.

    :param image: numpy.array
        Image data (single channel).
    :param regions: list
        A list of regions [(row_min, row_max, col_min, col_max),...]
    :param desc: LocalDescriptor
        Descriptor used for extracting the features.
    :return: dict
        A dictionary with two elements:
            <name of the descriptor>: list
            'regions': list
        The first list contains the feature descriptors.
        The second list contains the corresponding window positions
        (identical with the input regions list)

    See also: grow_bag_from_new_image
    """

    w_offset = (0, 0)
    if isinstance(desc, HaarLikeDescriptor):
        # this one works on integral images
        image = intg_image(image)
        # the sliding window should also be increased by 1:
        w_offset = (1, 1)

    bag = []

    for w in regions:
        # adjust if needed:
        r = (w[0], w[1] + w_offset[1], w[2], w[3] + w_offset[0])
        bag.append(desc.compute(image[r[0]:r[1], r[2]:r[3]]))

    return {desc.name: bag, 'regs': regions}


def grow_bag_from_images(img_list, desc_list, w_size, img_n):
    big_bag = None
    inames = []
    for img in img_list:
        im = skimage.io.imread(img)
        im = img_as_float(im)
        bag = None
        for d in desc_list:
            if bag is None:
                bag = grow_bag_from_new_image(im, d, w_size, img_n)
            else:
                bag[d.name] = grow_bag_with_new_features(im, bag['regs'], d)[d.name]

        if big_bag is None:
            big_bag = bag
        else:
            for k in big_bag.keys():
                big_bag[k].extend(bag[k])

        inames.extend(len(bag['regs'])*[img])  # add the name of the image

    big_bag['image'] = inames

    return big_bag


def read_bag(infile, desc_name):
    """
    Read a bag of features from an external file. It is assumed that the
    file contains values corresponding to a single type of descriptors,
    since no information distinguishing the descriptors is stored in the
    file.
    
    :param infile: string
        file name
    :param desc_name: string
        the name of the descriptor whose values are read
    :return: dict
        a bag of features with 'regs' and <desc_name> keys
    """

    with open(infile, 'r') as f:
        x = np.loadtxt(f)

    # x contains, as a matrix, all the local deascriptors, supposedly of the
    # same type. The following data is in x:
    # -first 4 columns in x correspond to the rectangular region on which
    #  the feature was computed
    # -the remaining columns correspond to descriptor values
    # -each row corresponds to a patch in the image (at coordinates given
    #  by the first 4 values)
    ndesc, nfeat = x.shape
    nfeat -= 4  # first 4 values are coords

    # put everything in a regular B-o-T dictionary (as above):
    regs = []
    desc = []
    for i in range(ndesc):
        regs.append(np.int32(x[i, 0:4]))
        desc.append(x[i, 4:])

    return {desc_name: desc, 'regs': regs}


def build_codebook_kmeans_online(bag, codebook_size, desc_names, standardize=False):
    desc = [np.array(bag[dn_]) for dn_ in desc_names]                  # ensures a strict ordering

    X = np.hstack(desc)                                                # put all feature vectors in an array
    if standardize:
        # make sure each variable (column) is mean-centered and has unit standard deviation
        Xm = np.mean(X, axis=0)
        Xs = np.std(X, axis=0)
        Xs[np.isclose(Xs, 1e-16)] = 1.0
        X = (X - Xm) / Xs
    
    if codebook_size is None:
        # try to estimate a suitable codebook size based on gap statistic:
        codebook_size,_ = gap(X, Ks=np.linspace(start=10, stop=100, num=10, dtype=np.int32),
                              Wstar=None, B=20)
        print("Best codebook size:", codebook_size)
            
    rng = np.random.RandomState(0)
    vq = MiniBatchKMeans(n_clusters=codebook_size, random_state=rng,
                         batch_size=500, compute_labels=True, verbose=True)   # vector quantizer

    vq.fit(X)
    
    return vq