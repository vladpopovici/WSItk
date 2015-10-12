#!/usr/bin/env python2

#
# wsi_bot_apply2
#
# Version 2 of Bag-Of_things:
#
# -uses OpenCV for faster operation - but different local descriptors than in the 1st version;
# -uses annotation files for defining the regions from where the descriptors are to be
#  extracted

from __future__ import (absolute_import, division, print_function, unicode_literals)

__version__ = 0.1
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import numpy as np
from scipy.linalg import norm
import skimage.draw
import skimage.io
from skimage.exposure import equalize_adapthist, rescale_intensity, adjust_gamma
import cv2
import cv2.xfeatures2d
from sklearn.cluster import MiniBatchKMeans

from stain.he import rgb2he
from util.storage import ModelPersistence


def main():
    p = opt.ArgumentParser(description="""
            Classifies regions of an image (based on SURF) using a pre-built model (codebook).
            """)
    p.add_argument('in_file', action='store', help='image file name')
    p.add_argument('out_file', action='store', help='file to store the resulting classification')
    p.add_argument('model', action='store', help='file containing the model')
    p.add_argument('-a', '--annot', action='store', help='annotation file name', default=None)
    p.add_argument('-t', '--threshold', action='store', type=int, default=5000,
                   help='Hessian threshold for SURF features.')
    p.add_argument('-x', action='store', help='image name with patches classified', default=None)
    p.add_argument('-v', '--verbose', action='store_true', help='verbose?')
    
    args = p.parse_args()
    th = args.threshold
    
    if args.verbose:
        print("Image:", args.in_file)
        
    img = cv2.imread(args.in_file)
    mask = None
    
    with ModelPersistence(args.model, 'r', format='pickle') as mp:
        codebook = mp['codebook']
        Xm = mp['shift']
        Xs = mp['scale']
        standardize = mp['standardize']
        avg_dist = mp['avg_dist_to_centroid']
        sd_dist = mp['stddev_dist_to_centroid']
        
    if args.annot is not None:
        coords = np.fromfile(args.annot, dtype=int, sep=' ')  # x y - values
        coords = np.reshape(coords, (coords.size/2, 2), order='C')
        # get the bounding box:
        xmin, ymin = coords.min(axis=0)
        xmax, ymax = coords.max(axis=0)
        img = img[ymin:ymax+3, xmin:xmax+3, :]            # keep only the region of interest

        if args.verbose:
            print("\t...building mask")
        
        mask = np.zeros(img.shape[0:2], dtype=np.uint8)
        r, c = skimage.draw.polygon(coords[:,1]-ymin, coords[:,0]-xmin) # adapt to new image...
        mask[r,c] = 1                                         # everything outside the region is black

    if args.verbose:
        print("\t...H&E extraction")

    img_h, _ = rgb2he(img, normalize=True)                # get the H- component
    img_h = equalize_adapthist(img_h)
    img_h = rescale_intensity(img_h, out_range=(0,255))
    
    # make sure the dtype is right for image and the mask: OpenCV is sensitive to data type
    img_h = img_h.astype(np.uint8)

    if mask is not None:
        img_h *= mask
    
    if args.verbose:
        print("\t...feature detection and computation")
    
    feat = cv2.xfeatures2d.SURF_create(hessianThreshold=th)
    keyp, desc = feat.detectAndCompute(img_h, mask)
    
    if args.verbose:
        print("\t...", str(len(keyp)), "features extracted")
        
    X = np.hstack(desc)
    X = np.reshape(X, (len(desc), desc[0].size), order='C')
    if standardize:
        # make sure each variable (column) is mean-centered and has unit standard deviation
        X = (X - Xm) / Xs
        
    if args.verbose:
        print("\t...classification")
        
    # instead of the following, allow for "no label":
    y0 = codebook.predict(X).tolist()
    y = np.zeros(X.shape[0], dtype=np.int) - 1
    d = np.zeros((X.shape[0], codebook.cluster_centers_.shape[0]))
    
    for k in range(0, codebook.cluster_centers_.shape[0]):
        d[:, k] = np.linalg.norm(X - codebook.cluster_centers_[k, :], axis=1)

    for i in range(0, d.shape[0]):
        # find the closest centroid among those that have a distance < 3*SD
        j = np.where(d[i, :] < avg_dist + 3.0*sd_dist)[0]
        if np.any(j):
            y[i] = j[np.argmin(d[i, j])]    # the label of the closest centroid
    
    #if np.any(y < 0):
    #    y = y[y >= 0]
        
    if args.verbose:
        print("\t...of", str(X.shape[0]), "patches,", str(y.size), "where assigned a label")
    with open(args.out_file, mode='w') as fout:
        for k in range(len(y)):
            s = '\t'.join([str(np.round(keyp[k].pt[0])), str(np.round(keyp[k].pt[1])),
                           str(np.round(keyp[k].size)), str(y[k]), str(y0[k])]) + '\n'
            fout.write(s)

    if args.x is not None:
        # construct a representation of the image based on the class labels
        img = adjust_gamma(img, 0.2)             # dim the image
        for k in range(len(y)):
            x, y = keyp[k].pt
            x = int(np.round(x))
            y = int(np.round(y))
            r = int(np.round(keyp[k].size))
            img[y-int(r/2):y+int(r/2), x-int(r/2):x+int(r/2), :] = (10, (10+2*k)%256, k%256)
        cv2.imwrite(args.x, img)
    
if __name__ == '__main__':
    main()