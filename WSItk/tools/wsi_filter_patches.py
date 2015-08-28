# -*- coding: utf-8 -*-
"""
WSI_FILTER_PATCHES

Given a mask (binary image, with non-zero pixels indicating the object of interest)
and a set of patches, filter out those that do not overlap (to a specified degree)
with the mask. Returns only patches of interest.

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__author__ = 'Vlad Popovici'
__version__ = 0.1


import argparse as opt
import fileinput
import skimage.io


def main():
    p = opt.ArgumentParser(description="""
    Filters out patches not overlapping with a support region (mask). The list of
    patches can be read from a specified file or from STDIN.
    """)

    p.add_argument('mask', action='store', help='mask image file name')
    p.add_argument('-p', '--prop', action='store', type=float,
                   help='minimum proportion of the patch that must be inside the mask so it is accepted',
                   default=0.5)

    args, unk = p.parse_known_args()

    if args.prop <= 0 or args.prop > 1.0:
        raise RuntimeError("The proportion must be a number > 0.0 and <= 1.0")

    # read the mask:
    mask = skimage.io.imread(args.mask)
    mask[mask > 0] = 1                                    # make sure is 0/1-valued
    for l in fileinput.input(unk):
        l = l.strip()
        r = [int(x_) for x_ in l.split()[:4]]     # the first 4 values define a patch
        a0 = (r[1] - r[0]) * (r[3] - r[2])                # area of the patch
        a1 = mask[r[0]:r[1], r[2]:r[3]].sum()             # intersection patch and mask
        if float(a1) / float(a0) >= args.prop:
            print(l)

    return

if __name__ == '__main__':
    main()
