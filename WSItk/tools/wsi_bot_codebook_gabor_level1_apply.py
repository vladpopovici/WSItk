# -*- coding: utf-8 -*-
"""
TOOLS.WSI_BOT_CODEBOOK_GABOR_LEVEL1: bag-of-things from an image series; level-1 with Gabor-based level-0 codebook.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__version__ = 0.1
__author__ = 'Vlad Popovici'

import argparse as opt
import numpy as np

from util.storage import ModelPersistence
from util.math import dist

from joblib import *


def worker_nearest_medoid(p, Q):
    # finds the closest medoid in Q to p (row index)
    r = [dist.chisq(p, q) for q in Q]
    return np.argmin(r)


def main():
    p = opt.ArgumentParser(description="""
            Recodes an image based on Level 1 bag-of-things model. This means that the original image
            has been already re-coded in terms of a Level 0 bag-of-things model and the resulting
            coding is fed into this program. The input data is a file containing the region coordinates
            and the histogram of level 0 features for each window in the original image that is to be
            recoded. The output will preserve the same structure: regions and assigned label.
            """)

    p.add_argument('in_data', action='store', help='level 0-coded image')
    p.add_argument('out_data', action='store', help='resulting level 1-coded image')
    p.add_argument('l1_model', action='store', help='level-0 codebook model file')

    args = p.parse_args()

    with ModelPersistence(args.l1_model, 'r', format='pickle') as mp:
        l1_model = mp

    with ModelPersistence(args.in_data, 'r', format='pickle') as d:
        data = d['bag_l1']


    block_codes = \
        Parallel(n_jobs=cpu_count()) \
        ( delayed(worker_nearest_medoid)(p, l1_model['codebook']['cluster_centers_']) for p in data['hist_l0'] )

    with ModelPersistence(args.out_data, 'c', format='pickle') as d:
        d['l1_codes'] = block_codes
        d['regs'] = data['regs']

    return
# end main()

if __name__ == '__main__':
    main()
