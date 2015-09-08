#!/usr/bin/env python2

#
# wsi_bot_codebook2
#
# Version 2 of codebook construction:
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
import skimage.draw
import skimage.io
import cv2
import cv2.xfeatures2d
from sklearn.cluster import MiniBatchKMeans

from stain.he import rgb2he
from util.storage import ModelPersistence


def main():
    p = opt.ArgumentParser(description="""
            Extracts features from annotated regions and constructs a codebook of a given size.
            """)
    p.add_argument('in_file', action='store', help='a file with pairs of image and annotation files')
    p.add_argument('out_file', action='store', help='resulting model file name')
    p.add_argument('codebook_size', action='store', help='codebook size', type=int)
    p.add_argument('-t', '--threshold', action='store', type=int, default=5000,
                   help='Hessian threshold for SURF features.')
    p.add_argument('-s', '--standardize', action='store_true', default=False,
                   help='should the features be standardized before codebook construction?')
    p.add_argument('-x', action='store', help='save the image patches closes to the code blocks?')
    p.add_argument('-v', '--verbose', action='store_true', help='verbose?')
    
    
    args = p.parse_args()
    th = p.threshold
    
    all_key_points, all_descriptors, all_image_names = [], [], []
    all_roi = []
    with open(args.in_file, mode='r', encoding="ascii", errors="surrogateescape") as fin:
        for l in fin.readlines():
            img_file, annot_file = [z_ for z_ in l.split()][0:2]  # file names: image and its annotation
            
            if args.verbose:
                print("Image:", img_file)
                
            img = cv2.imread(img_file)
            coords = np.fromfile(annot_file, dtype=int, sep=' ')  # x y - values
            coords = np.reshape(coords, (coords.size/2, 2), order='C')
            # get the bounding box:
            xmin, ymin = coords.min(axis=0)
            xmax, ymax = coords.max(axis=0)

            if args.verbose:
                print("\t...H&E extraction")

            img = img[ymin:ymax+1, xmin:xmax+1, :]                # keep only the region of interest
            img_h, _ = rgb2he(img, normalize=True)                # get the H- component
            img_h = equalize_adapthist(img_h)
            img_h = rescale_intensity(img_h, out_range=(0,255))
            
            # make sure the dtype is right for image and the mask: OpenCV is sensitive to data type
            img_h = img_h.astype(np.uint8)

            if args.verbose:
                print("\t...building mask")
                
            mask = np.zeros(img_h.shape, dtype=np.uint8)
            r, c = skimage.draw.polygon(coords[:,1]-ymin, coords[:,0]-xmin) # adapt to new image...
            mask[r,c] = 1                                         # everything outside the region is black
            
            if args.verbose:
                print("\t...feature detection and computation")
                
            feat = cv2.xfeatures2d.SURF_create(hessianThreshold=th)
            keyp, desc = feat.detectAndCompute(img_h, mask)
            
            if args.verbose:
                print("\t...", str(len(keyp)), "features extracted")
                
            all_descriptors.extend(desc)
            if args.x:
                # only needed if saving patches:
                all_key_points.extend(keyp)
                all_image_names.extend([img_file] * len(keyp))
                all_roi.extend([(xmin, xmax, ymin, ymax)] * len(keyp))
        # end for
    
    if args.verbose:
        print("\t...k-means clustering")
        
    X = np.hstack(all_descriptors)
    
    if args.standardize:
        # make sure each variable (column) is mean-centered and has unit standard deviation
        Xm = np.mean(X, axis=0)
        Xs = np.std(X, axis=0)
        Xs[np.isclose(Xs, 1e-16)] = 1.0
        X = (X - Xm) / Xs
        
    rng = np.random.RandomState(0)
    vq = MiniBatchKMeans(n_clusters=args.codebook_size, random_state=rng,
                         batch_size=500, compute_labels=False, verbose=True)   # vector quantizer

    vq.fit(X)

    with ModelPersistence(args.out_file, 'c', format='pickle') as d:
        d['codebook'] = vq
        d['shift'] = Xm
        d['scale'] = Xs
        d['standardize'] = args.standardize

    if args.x:
        # find the closest patches to each centroid:
        idx = np.zeros(args.codebook_size)
        d = np.zeros(X.shape[0])
        for k in range(0, args.codebook_size):
            for i in range(0, X.shape[0]):
                d[i] = norm(X[i,:]- vq.cluster_centers_[k,:])
            idx[k] = d.argmin()        # the index of the closest patch to k-th centroid
        for k in range(0, args.codebook_size):
            x, y = all_key_points[idx[k]].pt
            x = int(np.round(x))
            y = int(np.round(y))
            r = all_key_points[idx[k]].size   # diameter of the region
            img = cv2.imread(all_image_names[k])
            patch = img[y+all_roi[k][2]-int(r/2):y+all_roi[k][2]+int(r/2),
                        x+all_roi[k][0]-int(r/2):x+all_roi[k][0]+int(r/2), :]
            cv2.imwrite('codeblock_'+str(k)+'.png', patch)
            
        
        
    return True


if __name__ == '__main__':
    main()
    