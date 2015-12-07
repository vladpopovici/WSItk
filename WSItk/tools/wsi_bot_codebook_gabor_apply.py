# -*- coding: utf-8 -*-
"""
WSI_BOT_CODEBOOK_GABOR_APPLY

Assigns all patches in an image to one of the clusters in the codebook (Gabor features).

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
from skimage.util import img_as_bool
from skimage.exposure import equalize_adapthist, rescale_intensity

from util.storage import ModelPersistence
from util.configs import read_local_descriptors_cfg
from util.explore import sliding_window_on_regions

from descriptors.basic import *
from descriptors.txtgrey import *
from stain.he import rgb2he

from joblib import *

def main():
    p = opt.ArgumentParser(description="""
    Assigns the regions of an image to the clusters of a codebook.
    """)
    p.add_argument('image', action='store', help='image file name')
    p.add_argument('model', action='store', help='model file name')
    p.add_argument('out_file', action='store', help='results file name')
    p.add_argument('-r', '--roi', action='store', nargs=4, type=int,
                   help='region of interest from the image as: row_min row_max col_min col_max',
                   default=None)
    args = p.parse_args()

    wsize = 32
    tmp  = np.array([0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0],
        dtype=np.double)
    tmp2 = np.array([3.0 / 4.0, 3.0 / 8.0, 3.0 / 16.0], dtype=np.double)
    tmp3 = np.array([1.0, 2 * np.sqrt(2.0)], dtype=np.double)

    desc = GaborDescriptor(theta=tmp, freq=tmp2, sigma=tmp3)

    image = skimage.io.imread(args.image)
    if image.ndim == 3:
        im_h, _ = rgb2he(image, normalize=True)
        im_h = equalize_adapthist(im_h)
        im_h = rescale_intensity(im_h, out_range=(0,255))
        im_h = im_h.astype(np.uint8)
        image = im_h
        im_h = None

    if args.roi is None:
        roi = (0, image.shape[0]-1, 0, image.shape[1]-1)
    else:
        roi = args.roi

    with ModelPersistence(args.model, 'r', format='pickle') as mp:
        codebook = mp['codebook']
        avg_dist = None
        sd_dist = None
        if 'avg_dist_to_centroid' in mp:
            avg_dist = mp['avg_dist_to_centroid']
        if 'stddev_dist_to_centroid' in mp:
            sd_dist = mp['stddev_dist_to_centroid']


    itw = sliding_window_on_regions(image.shape, [tuple(roi)], (wsize,wsize), step=(wsize,wsize))
    wnd = []
    labels = []
    dists = []
    buff_size = 100                  # every <buff_size> patches we do a classification
    X = np.zeros((buff_size, codebook.cluster_centers_[0].shape[0]))

    k = 0
    for r in itw:
        # adjust if needed:
        r2 = (r[0], r[1], r[2], r[3])
        wnd.append(r2)
        X[k,:] = desc.compute(image[r[0]:r[1], r[2]:r[3]])
        k += 1
        if k == buff_size:
            y = codebook.predict(X)
            Z = codebook.transform(X)
            labels.extend(y.tolist())
            dists.extend(Z[np.arange(buff_size), y].tolist())  # get the distances to the centroids of the assigned clusters
            k = 0                      # reset the block

    if k != 0:
        # it means some data is accumulated in X but not yet classified
        y = codebook.predict(X[0:k,])
        Z = codebook.transform(X[0:k,])
        labels.extend(y.tolist())
        dists.extend(Z[np.arange(k), y].tolist())  # get the distances to the centroids of the assigned clusters

    # save data
    with open(args.out_file, 'w') as f:
        n = len(wnd)                       # total number of descriptors of this type
        for k in range(n):
            s = '\t'.join([str(x_) for x_ in wnd[k]]) + '\t' + str(labels[k]) + \
                '\t' + str(dists[k]) + '\n'
            f.write(s)

if __name__ == '__main__':
    main()
