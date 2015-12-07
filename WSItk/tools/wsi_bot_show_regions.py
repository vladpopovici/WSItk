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
import numpy as np

from util.storage import ModelPersistence
from util.visualization import enhance_patches

__author__ = 'vlad'
__version__ = 0.1


def main():
    p = opt.ArgumentParser(description="""
    Emphasizes the patches with a given code (from BoT) by reducing the contrast of the rest of the image.
    """
    )
    p.add_argument('image', action='store', help='image file name')
    p.add_argument('res_image', action='store', help='name of the resulting image')
    p.add_argument('bot_result', action='store', help='a file with BoT coding for regions')
    p.add_argument('bot_code', action='store', help='the code of the regions to be emphasized', type=int)
    p.add_argument('-g', '--gamma', action='store', nargs=1, type=float,
                   help='the gamma level of the background regions',
                   default=0.2)
    args = p.parse_args()

    img = skimage.io.imread(args.image)
    regs = []
    with ModelPersistence(args.bot_result, 'r', format='pickle') as d:
        block_codes = d['l1_codes']
        regs = d['regs']

    #print(block_codes)
    #print(args.bot_code)
    # filter regions of interest:
    roi = [ regs[k] for k in np.where(np.array(block_codes, dtype=np.int) == args.bot_code)[0] ]

    #print(roi)

    img = enhance_patches(img, roi, _gamma=args.gamma)

    skimage.io.imsave(args.res_image, img)

    return


if __name__ == '__main__':
    main()