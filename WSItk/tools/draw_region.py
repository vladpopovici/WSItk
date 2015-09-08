# -*- coding: utf-8 -*-
"""
DRAW_REGION

Draw the contour of a polygonal region (read from STDIN).

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__author__ = 'Vlad Popovici'
__version__ = 0.1


import argparse as opt
import fileinput
import skimage.io
import skimage.draw
import numpy as np

def main():
    p = opt.ArgumentParser(description="""
    Draw a polygon in the image.
    """)

    p.add_argument('image', action='store', help='image file name')
    p.add_argument('-o', '--overwrite', action='store', type=bool,
                   help='should the result be written in the same file?',
                   default=False)

    args, unk = p.parse_known_args()

    outfile = args.image if args.overwrite else "result.png"
    
    # read the image:
    img = skimage.io.imread(args.image)
    
    x, y = [], []
    
    for l in fileinput.input(unk):
        l = l.strip()
        r = [int(float(x_)) for x_ in l.split()]   # x, y - values
        x.append(r[0])
        y.append(r[1])
        
    r, c = skimage.draw.polygon(np.array(y), np.array(x))
    img[r,c,:] = 255
        
    skimage.io.imsave(outfile, img)
    
    return


if __name__ == '__main__':
    main()
    
