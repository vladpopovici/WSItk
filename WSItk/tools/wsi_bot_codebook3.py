#!/usr/bin/env python2

#
# wsi_bot_codebook3
#
# Version 3 of codebook construction:
#
# -uses OpenCV for faster operation - but different local descriptors than in the 1st version;
# -uses annotation files for defining the regions from where the descriptors are to be
#  extracted
# - try to optimize the codebook with respect to some class labels

from __future__ import (absolute_import, division, print_function, unicode_literals)

__version__ = 0.1
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import numpy as np
import numpy.linalg
from scipy.stats import ttest_ind

import skimage.draw
import skimage.io
from skimage.exposure import equalize_adapthist, rescale_intensity
import cv2
import cv2.xfeatures2d
from sklearn.cluster import MiniBatchKMeans
from sklearn.lda import LDA

from stain.he import rgb2he
from util.storage import ModelPersistence

def find_in_list(_value, _list):
    """
    Returns the indexes of all occurrences of value in a list.
    """
    return np.array([i for i, v in enumerate(_list) if v == _value], dtype=int)


def main():
    p = opt.ArgumentParser(description="""
            Extracts features from annotated regions and constructs a codebook of a given size.
            """)
    p.add_argument('in_file', action='store', help='a file with image file, annotation file and label (0/1)')
    p.add_argument('out_file', action='store', help='resulting model file name')
    #p.add_argument('codebook_size', action='store', help='codebook size', type=int)
    p.add_argument('-t', '--threshold', action='store', type=int, default=5000,
                   help='Hessian threshold for SURF features.')
    p.add_argument('-s', '--standardize', action='store_true', default=False,
                   help='should the features be standardized before codebook construction?')
    p.add_argument('-v', '--verbose', action='store_true', help='verbose?')
    
    args = p.parse_args()
    th = args.threshold
    
    all_image_names, all_descriptors = [], []
    all_roi = []
    y = []
    unique_image_names = []
    with open(args.in_file, mode='r') as fin:
        for l in fin.readlines():
            l = l.strip()
            if len(l) == 0:
                break
            img_file, annot_file, lbl = [z_ for z_ in l.split()][0:3]  # file names: image and its annotation and label
            y.append(int(lbl))
            
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

            img = img[ymin:ymax+2, xmin:xmax+2, :]                # keep only the region of interest
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
            
            img_h *= mask
            feat = cv2.xfeatures2d.SURF_create(hessianThreshold=th)
            keyp, desc = feat.detectAndCompute(img_h, mask)
            
            if args.verbose:
                print("\t...", str(len(keyp)), "features extracted")
                
            all_descriptors.extend(desc)
            all_image_names.extend([img_file] * len(keyp))
            unique_image_names.append(img_file)            
        # end for
            
    X = np.hstack(all_descriptors)
    X = np.reshape(X, (len(all_descriptors), all_descriptors[0].size), order='C')
    if args.standardize:
        # make sure each variable (column) is mean-centered and has unit standard deviation
        Xm = np.mean(X, axis=0)
        Xs = np.std(X, axis=0)
        Xs[np.isclose(Xs, 1e-16)] = 1.0
        X = (X - Xm) / Xs
    
    y = np.array(y, dtype=int)
    
    rng = np.random.RandomState(0)
    acc = []                           # will keep accuracy of the classifier
    vqs = []                           # all quantizers, to find the best
    for k in np.arange(10, 121, 10):
        # Method:
        # -generate a codebook with k codewords
        # -re-code the data
        # -compute frequencies
        # -estimate classification on best 10 features
        
        if args.verbose:
            print("\nK-means clustering (k =", str(k), ")")
            print("\t...with", str(X.shape[0]), "points")
        
        #-codebook and re-coding
        vq = MiniBatchKMeans(n_clusters=k, random_state=rng,
                         batch_size=500, compute_labels=True, verbose=False)   # vector quantizer
        vq.fit(X)
        vqs.append(vq)
        
        #-codeword frequencies
        frq = np.zeros((len(unique_image_names), k))
        for i in range(vq.labels_.size):
            frq[unique_image_names.index(all_image_names[i]), vq.labels_[i]] += 1.0

        for i in range(len(unique_image_names)):
            if frq[i, :].sum() > 0:
                frq[i, :] /= frq[i, :].sum()

        if args.verbose:
            print("...\tfeature selection (t-test)")
        pv = np.ones(k)
        for i in range(k):
            _, pv[i] = ttest_ind(frq[y == 0, i], frq[y == 1, i])
        idx = np.argsort(pv)         # order of the p-values
        if args.verbose:
            print("\t...classification performance estimation")
        clsf = LDA(solver='lsqr', shrinkage='auto').fit(frq[:,idx[:10]], y) # keep top 10 features
        acc.append(clsf.score(frq[:, idx[:10]], y))
    
    acc = np.array(acc)
    k = np.arange(10, 121, 10)[acc.argmax()]  # best k
    if args.verbose:
        print("\nOptimal codebook size:", str(k))

    # final codebook:
    vq = vqs[acc.argmax()]

    # compute the average distance and std.dev. of the points in each cluster:
    avg_dist = np.zeros(k)
    sd_dist = np.zeros(k)
    for k in range(0, k):
        d = numpy.linalg.norm(X[vq.labels_ == k, :] - vq.cluster_centers_[k, :], axis=1)
        avg_dist[k] = d.mean()
        sd_dist[k] = d.std()

    with ModelPersistence(args.out_file, 'c', format='pickle') as d:
        d['codebook'] = vq
        d['shift'] = Xm
        d['scale'] = Xs
        d['standardize'] = args.standardize
        d['avg_dist_to_centroid'] = avg_dist
        d['stddev_dist_to_centroid'] = sd_dist

    return True


if __name__ == '__main__':
    main()
    