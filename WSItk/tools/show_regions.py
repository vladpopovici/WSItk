# -*- coding: utf-8 -*-
"""
SHOW_REGIONS

Emphasizes some regions in the image, by decreasing the importance of the rest.

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

import argparse as opt
import skimage.io

from util.visualization import enhance_patches

__author__ = 'vlad'
__version__ = 0.1


def main():
    p = opt.ArgumentParser(description="""
    Emphasizes some regions of an image by reducing the contrast of the rest of the image.
    """,
             epilog="""
    The regions are specified in an external file where each (rectangular) region is given in
    terms of corner coordinates - row min, row max, col min, col max - and a (numeric) label.
    The user must specify the label of the regions to be emphasized.
    """)
    p.add_argument('image', action='store', help='image file name')
    p.add_argument('result', action='store', help='result image file')
    p.add_argument('regions', action='store', help='file with the regions of interest')
    p.add_argument('label', action='store', nargs='+', type=int,
                   help='one or more labels (space-separated) of the regions to be emphasized')
    p.add_argument('-g', '--gamma', action='store', nargs=1, type=float,
                   help='the gamma level of the background regions',
                   default=0.2)
    args = p.parse_args()
    img_file = args.image
    res_file = args.result
    reg_file = args.regions
    labels = args.label
    gam = args.gamma

    img = skimage.io.imread(img_file)
    regs = []
    
    with open(reg_file, 'r') as f:
        lines = f.readlines()
        for l in lines:
            r = [int(x_) for x_ in l.strip().split()[:-1]]
            if r[4] in labels:
                regs.append(r[0:4])

    img = enhance_patches(img, regs, _gamma=gam)

    skimage.io.imsave(res_file, img)

    return


if __name__ == '__main__':
    main()