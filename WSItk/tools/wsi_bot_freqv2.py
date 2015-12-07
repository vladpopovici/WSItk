# -*- coding: utf-8 -*-
"""
WSI_BOT_FREQV2

After an image has been recoded - i.e. all patches of interest were assign to the
corresponding cluster - this program will compute the code block frequency vector.

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__author__ = 'Vlad Popovici'
__version__ = 0.1


import argparse as opt

import skimage.io
from skimage.measure import *
from skimage.exposure import rescale_intensity

import numpy as np
import scipy.stats as st


def main():
    p = opt.ArgumentParser(description="""
    Compute the code block frequency vector and, optionally, produce a pseudo
    image with pixel intensitites indicating the local label.

    The result is printed to STDOUT.
    """)

    p.add_argument('data', action='store', help='data file with patch labels')
    p.add_argument('nclust', action='store', type=int, help='number of clusters in the model')
    p.add_argument('-p', '--pseudo', action='store', help='name of the pseudo-image file', default=None)

    args = p.parse_args()

    v = np.zeros((6*args.nclust), dtype=np.float64)
    r = np.loadtxt(args.data)      # read all data: 4 columns windows coords, then label and distance

    # find the extent of the image covered and local neighborhood size:
    rmin = r[:,0].min()
    rmax = r[:,1].max()
    cmin = r[:,2].min()
    cmax = r[:,3].max()
    wnd  = (r[0,1] - r[0,0], r[0,3] - r[0,2])

    nwnd = (int((rmax - rmin)/wnd[0]), int((cmax - cmin)/wnd[1]))

    # counts:
    for k in r[:,4]:
        v[int(k)] += 1.0

    # construct a pseudo-image with pixel intensities given by the patch label:
    im = np.zeros(nwnd, dtype=np.uint64)
    i  = ((r[:,0] - rmin) / wnd[0]).round().astype(np.int64)
    j  = ((r[:,2] - cmin) / wnd[1]).round().astype(np.int64)
    x  = r[:,4].astype(np.uint64)
    im[i,j] = x

    # for each possible label (0..nclust-1) compute a few statistical descriptors:
    # -median area of objects with the same label
    # -maximum area
    # -ratio of the maximal area of an object and total area of regions with the same label
    # -skewness of area values
    # -average compactness of ...

    for l in np.arange(args.nclust):
        b = (im == l).astype(np.int)  # binary mask
        if b.sum() == 0:
            continue                  # no patch with label l was found...
        obj, nobj = label(b, connectivity=2, return_num=True)
        props = regionprops(obj)
        a = np.array([p.area for p in props])
        p = np.array([p.perimeter for p in props])
        # 1-pixel objects have null perimeter, fix it to be 1:
        p[p == 0] += 1
        c = p**2 / np.array([p.area for p in props])
        v[l +   args.nclust] = np.median(a)
        v[l + 2*args.nclust] = a.max()
        v[l + 3*args.nclust] = a.max() / a.sum()
        v[l + 4*args.nclust] = st.skew(a)
        v[l + 5*args.nclust] = np.mean(c)

    if args.pseudo is not None:
        im = rescale_intensity(im, out_range=(0,255))
        im = im.astype(np.uint8)
        skimage.io.imsave(args.pseudo, im)

    print(' '.join(["{:.10f}".format(x_) for x_ in v]))

    return

if __name__ == '__main__':
    main()
