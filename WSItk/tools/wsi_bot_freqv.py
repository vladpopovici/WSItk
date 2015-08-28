# -*- coding: utf-8 -*-
"""
WSI_BOT_FREQV

After an image has been recoded - i.e. all patches of interest were assign to the
corresponding cluster - this program will compute the code block frequency vector.

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__author__ = 'Vlad Popovici'
__version__ = 0.1


import argparse as opt
import fileinput
import skimage.io
import numpy as np


def main():
    p = opt.ArgumentParser(description="""
    Compute the code block frequency vector. The list of patches can be read
    from a specified file or from STDIN, so the program accepts both forms:
    
    wsi_bot_freqv <nclusters> <filename>
    
    or
    
    wsi_bot_freqv <nclusters> < file_with_patches
    
    The result is printed to STDOUT.
    """)

    p.add_argument('nclust', action='store', type=int,
                   help='number of clusters in the model')
    
    args, unk = p.parse_known_args()
    v = np.zeros((args.nclust))
    
    for l in fileinput.input(unk):
        l = l.strip()
        # the first 4 values define a patch and the 5th the code block index
        # we care only about the index:
        k = int(l.split()[4])
        v[k] += 1

    v = v / v.sum()                    # get frequencies
    
    print(' '.join(["{:.10f}".format(x_) for x_ in v]))
    
    return

if __name__ == '__main__':
    main()
