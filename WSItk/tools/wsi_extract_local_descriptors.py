#!/usr/bin/env python2
"""
Extract local descriptors from images of H&E-stained pathology sections.
"""
from __future__ import (division, print_function, unicode_literals)

__version__ = 0.01
__author__ = 'vlad'

import os
import argparse as opt
# import xml.etree.ElementTree as ET
# from xml.dom import minidom

from descriptors.extract import get_gabor_desc
from descriptors.txtgrey import GaborDescriptor
from util.intensity import requantize
from stain.he import rgb2he
from segm.tissue import tissue_region_from_rgb

# from util.storage import ModelPersistence
import skimage.io


def main():
    p = opt.ArgumentParser(description="""
            Computes textural tissue descriptors from an RGB image (of an H&E slide).
            """)
    p.add_argument('img_file', action='store', help='RGB image file of an H&E slide')
    p.add_argument('out_file', action='store', default='descriptors.dat',
                   help='Name of the result file')

    # p.add_argument('model_file', action='store', help='Models file')
    p.add_argument('--scale', action='store', type=float, default=1.0,
                   help='Scale of the image at which the descriptors are computed (default: 1.0)')
    p.add_argument('--ngl', type=int, default=16, action='store',
                   help='Number of grey levels in H- and E-images (default: 16)')
    p.add_argument('--wsize', action='store', type=int, default=50,
                   help='Sliding window size (default: 50)')
    p.add_argument('--mask', action='store_true',
                   help='')


    args = p.parse_args()
    img_file = args.img_file
    # model_file = args.model_file
    n_grey_levels = args.ngl
    w_size = args.wsize
    scale = args.scale
    out_file = args.out_file

    base_name = os.path.basename(img_file).split('.')
    if len(base_name) > 1:             # at least 1 suffix .ext
        base_name.pop()                # drop the extension
        base_name = '.'.join(base_name)  # reassemble the rest of the list into file name

    img = skimage.io.imread(img_file)

    # with ModelPersistence(model_file, 'r', format='pickle') as d:
    #    rgb_models = d['models']

    img_h, img_e   = rgb2he(img, normalize=True)
    img_h          = requantize(img_h, nlevels=n_grey_levels, method='linear')
    img_e          = requantize(img_e, nlevels=n_grey_levels, method='linear')

    G = GaborDescriptor()
    if args.mask:
        mask, _ = tissue_region_from_rgb(img, _min_area=150)
        g_h = get_gabor_desc(img_h, G, w_size, scale, mask)
        g_e = get_gabor_desc(img_e, G, w_size, scale, mask)
    else:
        g_h = get_gabor_desc(img_h, G, w_size, scale)
        g_e = get_gabor_desc(img_e, G, w_size, scale)

    with open(out_file, 'w') as f:
        for d in g_h:
            f.write('\t'.join(str(x) for x in d))
            f.write('\n')
        for d in g_e:
            f.write('\t'.join(str(x) for x in d))
            f.write('\n')

    return

# end main


if __name__ == '__main__':
    main()