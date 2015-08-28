# -*- coding: utf-8 -*-
"""
WSI_BOT_CODEBOOK

Build a codebook for a given bag-of-things (-features).

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__version__ = 0.5
__author__ = 'Vlad Popovici'

import os
import sys
import argparse as opt
from ConfigParser import SafeConfigParser
import ast
import glob

import numpy as np
from sklearn.cluster import MiniBatchKMeans

from ml.gap import gap
from segm.bot import *
from util.storage import ModelPersistence

def main():
    p = opt.ArgumentParser(description="""
        Builds a codebook based on a set of local descriptors, previously
        computed.
        """)
    p.add_argument('config', action='store', help='a configuration file')
    args = p.parse_args()
    cfg_file = args.config

    parser = SafeConfigParser()
    parser.read(cfg_file)
    
    if not parser.has_section('codebook'):
        raise ValueError('"codebook" section is mandatory')
        
    codebook_size = ast.literal_eval(parser.get('codebook', 'size'))
    if isinstance(codebook_size, list):
        # expect 3 values:
        if len(codebook_size) != 3:
            raise ValueError('Wrong codebook size specification')
        codebook_size = np.linspace(*codebook_size, dtype=np.int32)
    elif isinstance(codebook_size, int):
        if codebook_size <= 0:
            raise ValueError('Wrong codebook size specification')
    else:
        raise ValueError('Wrong codebook size specification')

    verbose = False
    if parser.has_option('codebook', 'verbose'):
        verbose = parser.getboolean('codebook', 'verbose')
    standardize = True
    if parser.has_option('codebook', 'standardize_features'):
        standardize = parser.getboolean('codebook', 'standardize_features')

    result_file = 'output.dat'
    if parser.has_option('codebook', 'result'):
        result_file = parser.get('codebook', 'result')

    # Read the various features:
    big_bag = {}
    img_names = []                     # for each region, the image it belongs to
    all_regs = []                      # all regions
    descriptors = []

    for desc_name in ['gabor', 'haar', 'identity', 'stats', 'hist', 'hog', 'lbp']:
        if not parser.has_option('codebook', desc_name):
            continue
        feat_files = []
        with open(parser.get('codebook', desc_name), 'r') as f:
            feat_files = f.readlines()

        if len(feat_files) == 0:
            raise UserWarning('No files specified for ' + desc_name + ' feature.')

        if verbose:
            print('Reading', desc_name)
        descriptors.append(desc_name)

        desc_values = []               # all values for this descriptor will be concatenated in a single list
        for f in feat_files:
            f = f.strip()
            if len(f) == 0:
                continue
            if verbose:
                print('\t', f)
            bag = read_bag(f, desc_name)
            desc_values.extend(bag[desc_name])
            if len(big_bag) == 0:
                # since the image names and regions are the same (in the same order too) for all
                # feature types (gabor, haar,...) it makes sense to add them only once, when the "big_bag"
                # is still empty (for the 1st feature type read)
                img_names.extend([f] * len(bag[desc_name]))  # for each feature, add the image name
                all_regs.extend(bag['regs'])

        if len(big_bag) == 0:
            # 1st time store some values:
            big_bag['regs'] = all_regs
            big_bag['fname'] = img_names

        big_bag[desc_name] = desc_values

    if verbose:
        print("Read", len(descriptors), "feature type(s) with a total of", len(big_bag['regs']), "regions.")
        print('\nBuilding codebook:')

    desc = [np.array(bag[dn_]) for dn_ in descriptors]                 # ensures a strict ordering
    X = np.hstack(desc)                                                # put all feature vectors in an array
    Xm = np.zeros((X.shape[1],), dtype=np.float32)
    Xs = np.ones((X.shape[1],), dtype=np.float32)

    if standardize:
        # make sure each variable (column) is mean-centered and has unit standard deviation
        Xm = np.mean(X, axis=0)
        Xs = np.std(X, axis=0)
        Xs[np.isclose(Xs, 1e-16)] = 1.0
        X = (X - Xm) / Xs

    if not isinstance(codebook_size, int):
        # try to estimate a suitable codebook size based on gap statistic:
        codebook_size,_ = gap(X, Ks=codebook_size, Wstar=None, B=20)
        if verbose:
            print("\tBest codebook size:", codebook_size)

    rng = np.random.RandomState(0)
    vq = MiniBatchKMeans(n_clusters=codebook_size, random_state=rng,
                         batch_size=500, compute_labels=False, verbose=True)   # vector quantizer

    vq.fit(X)

    with ModelPersistence(result_file, 'c', format='pickle') as d:
        d['codebook'] = vq
        d['shift'] = Xm
        d['scale'] = Xs
        d['standardize'] = standardize

    return True


## MAIN
if __name__ == '__main__':
    main()
