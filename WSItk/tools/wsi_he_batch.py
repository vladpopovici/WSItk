#!/usr/bin/env python2
"""
Haematoxylin and Eosin staining is the most common staining used for pathology
slides. This program extracts the information (intensity) corresponding to each
of the stainings, from a RGB image.
"""
from __future__ import (division, print_function, unicode_literals)


__version__ = 0.01
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import numpy as np

import skimage.io
import skimage.exposure
import skimage.color

import glob
from joblib import *

from stain.he import rgb2he
from segm.tissue import tissue_region_from_rgb

def worker(img_name, eosine_flag, histeq_flag, pfx):
    if pfx is None:
        base_name = os.path.basename(img_name).split('.')
        if len(base_name) > 1:             # at least 1 suffix .ext
            base_name.pop()                # drop the extension
            base_name = '.'.join(base_name)  # reassemble the rest of the list into file name

        pfx = base_name

    img = skimage.io.imread(img_name)

    # produce a mask:
    img_gray = skimage.color.rgb2gray(img)
    mask = img_gray > 0
    img_gray = None

    mask2, _ = tissue_region_from_rgb(img, _min_area=300)
    
    if eosine_flag:
        img_h, img_e = rgb2he(img, normalize=True)

        if histeq_flag:
            img_h = skimage.exposure.equalize_adapthist(img_h)
            img_e = skimage.exposure.equalize_adapthist(img_e)

        img_h = skimage.exposure.rescale_intensity(img_h, out_range=(0,255))
        img_e = skimage.exposure.rescale_intensity(img_e, out_range=(0,255))

        img_h = img_h.astype(np.uint8)
        img_h[np.logical_not(mask)] = 0
        img_h[np.logical_not(mask2)] = 0
        img_e = img_e.astype(np.uint8)
        img_e[np.logical_not(mask)] = 0
        img_e[np.logical_not(mask2)] = 0

        skimage.io.imsave(pfx + '_h.png', img_h)
        skimage.io.imsave(pfx + '_e.png', img_e)
    else:
        img_h, _ = rgb2he(img, normalize=True)

        if histeq_flag:
            img_h = skimage.exposure.equalize_adapthist(img_h)

        img_h = skimage.exposure.rescale_intensity(img_h, out_range=(0,255))

        img_h = img_h.astype(np.uint8)
        img_h[np.logical_not(mask)] = 0
        img_h[np.logical_not(mask2)] = 0

        skimage.io.imsave(pfx + '_h.png', img_h)

    return


def main():
    p = opt.ArgumentParser(description="""
            Extracts the Haematoxylin and Eosine components from RGB images (of an H&E slide).
            """)
    p.add_argument('img_path', action='store', help='path to image files - all images in the folder will be used')
    p.add_argument('img_ext', action='store', help='extension of the image files (e.g. "jpg" or "png") - NO DOT!')

    p.add_argument('--prefix', action='store',
                   help='optional prefix for the result files: prefix_[h|e].type',
                   default=None)
    p.add_argument('--histeq', action='store_true',
                   help='requests for histogram equalization of the results')

    p.add_argument('-e', '--eosine', action='store_true', help='save the Eosine component as well')

    args = p.parse_args()

    img_files = glob.glob(args.img_path + '/*.' + args.img_ext)
    if len(img_files) == 0:
        return


    Parallel(n_jobs=cpu_count()) ( delayed(worker)(img_name, args.eosine, args.histeq, args.prefix) for img_name in img_files )

    return


if __name__ == '__main__':
    main()
