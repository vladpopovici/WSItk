# -*- coding: utf-8 -*-
"""
WSI_BOT_APPLY

Assigns all patches in an image to one of the clusters in the codebook.

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__author__ = 'Vlad Popovici'
__version__ = 0.01

import argparse as opt
from ConfigParser import SafeConfigParser
import ast

import numpy as np
import skimage.io

from util.storage import ModelPersistence
from util.configs import read_local_descriptors_cfg
from util.explore import sliding_window_on_regions
from util.misc import intg_image
from descriptors.txtgrey import HaarLikeDescriptor
from stain.he import rgb2he2

def main():
    p = opt.ArgumentParser(description="""
    Assigns the regions of an image to the clusters of a codebook.
    """)
    p.add_argument('image', action='store', help='image file name')
    p.add_argument('config', action='store', help='a configuration file')
    p.add_argument('-r', '--roi', action='store', nargs=4, type=int,
                   help='region of interest from the image as: row_min row_max col_min col_max',
                   default=None)
    args = p.parse_args()
    img_file = args.image
    cfg_file = args.config

    image_orig = skimage.io.imread(img_file)
    if image_orig.ndim == 3:
        im_h, _, _ = rgb2he2(image_orig)

    if args.roi is None:
        roi = (0, im_h.shape[0]-1, 0, im_h.shape[1]-1)
    else:
        roi = args.roi

    # Process configuration file:
    parser = SafeConfigParser()
    parser.read(cfg_file)

    if not parser.has_section('data'):
        raise RuntimeError('Section [data] is mandatory')
    wsize = (32, 32)
    if parser.has_option('data', 'window_size'):
        wsize = ast.literal_eval(parser.get('data', 'window_size'))

    if not parser.has_option('data', 'model'):
        raise RuntimeError('model file name is missing in [data] section')
    model_file = parser.get('data', 'model')
    with ModelPersistence(model_file, 'r', format='pickle') as mp:
        codebook = mp['codebook']
        Xm = mp['shift']
        Xs = mp['scale']
        standardize = mp['standardize']

    if parser.has_option('data', 'output'):
        out_file = parser.get('data', 'output')
    else:
        out_file = 'output.dat'

    descriptors = read_local_descriptors_cfg(parser)

    # For the moment, it is assumed tha only one type of local descriptors is
    # used - no composite feature vectors. This will change in the future but,
    # for the moment only the first type of descriptor in "descriptors" list
    # is used, and the codebook is assumed to be constructed using the same.

    desc = descriptors[0]

    print(img_file)
    print(wsize)
    print(roi[0], roi[1], roi[2], roi[3])


    w_offset = (0, 0)
    if isinstance(desc, HaarLikeDescriptor):
        # this one works on integral images
        image = intg_image(im_h)
        # the sliding window should also be increased by 1:
        w_offset = (1, 1)
        wsize = (wsize[0] + w_offset[0], wsize[1] + w_offset[1])
    else:
        image = im_h

    itw = sliding_window_on_regions(image.shape, [tuple(roi)], wsize, step=wsize)
    wnd = []
    labels = []
    buff_size = 10000                  # every <buff_size> patches we do a classification
    X = np.zeros((buff_size, codebook.cluster_centers_[0].shape[0]))
    k = 0
    if standardize:                    # placed here, to avoid testing inside the loop
        for r in itw:
            # adjust if needed:
            r2 = (r[0], r[1] - w_offset[1], r[2], r[3] - w_offset[0])
            wnd.append(r2)
            X[k,:] = desc.compute(image[r[0]:r[1], r[2]:r[3]])
            k += 1
            if k == buff_size:
                X = (X - Xm) / Xs
                labels.extend(codebook.predict(X).tolist())
                k = 0                      # reset the block
    else:
        for r in itw:
            # adjust if needed:
            r2 = (r[0], r[1] - w_offset[1], r[2], r[3] - w_offset[0])
            wnd.append(r2)
            X[k,:] = desc.compute(image[r[0]:r[1], r[2]:r[3]])
            k += 1
            if k == buff_size:
                labels.extend(codebook.predict(X).tolist())
                k = 0                      # reset the block

    if k != 0:
        # it means some data is accumulated in X but not yet classified
        if standardize:
            X[0:k+1,] = (X[0:k+1,] - Xm) / Xs
        labels.extend(codebook.predict(X[0:k+1,]).tolist())

    with open(out_file, 'w') as f:
        n = len(wnd)                       # total number of descriptors of this type
        for k in range(n):
            s = '\t'.join([str(x_) for x_ in wnd[k]]) + '\t' + str(labels[k]) + '\n'
            f.write(s)

if __name__ == '__main__':
    main()
