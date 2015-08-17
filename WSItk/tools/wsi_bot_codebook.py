# -*- coding: utf-8 -*-
"""
WSI_BOT_CODEBOOK

Build a codebook for a given bag-of-things (-features).

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__version__ = 0.01
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
    if isinstance(codebook_size, 'list'):
        # expect 3 values:
        if len(codebook_size) != 3:
            raise ValueError('Wrong codebook size specification')
        codebook_size = np.linspace(*codebook_size)
    elif isinstance(codebook_size, int):
        if codebook_size <= 0:
            raise ValueError('Wrong codebook size specification')
    else:
        raise ValueError('Wrong codebook size specification')
        
    # Read the various features:
    big_bag = {}
    
    for desc_name in ['gabor', 'haar', 'identity', 'stats', 'hist', 'hog', 'lbp']:
        if not parser.has_option('codebook', desc_name):
            continue
        feat_files = []
        with open(parser.get_option('codebook', desc_name), 'r') as f:
            feat_files = f.readlines()

        desc_values = []               # all values for this descriptor will be concatenated in a single list
        for f in feat_files:
            f = f.strip()
            if len(f) == 0:
                continue
            bag = read_bag(f, desc_name)
            desc_values.extend(bag[desc_name])

        big_bag[desc_name] = desc_values
        
    return big_bag


## MAIN
if __name__ == '__main__':
    main()
