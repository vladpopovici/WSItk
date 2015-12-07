# -*- coding: utf-8 -*-
"""
PPDIST: parallel computation of pairwise distances.

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__version__ = 0.1
__author__ = 'Vlad Popovici'

import argparse as opt
import numpy as np
from joblib import *
import itertools

from util.math import dist


def worker_chisq_M(p, Q):
    r = np.zeros(Q.shape[0])
    for i in np.arange(Q.shape[0]):
        r[i] = dist.chisq(p, Q[i,:])
    return r.tolist()


def main():
    p = opt.ArgumentParser(description="""
            Computes pairwise distances for large data matrices.
            """)

    p.add_argument('data', action='store', help='data file (produced by numpy.save)')
    p.add_argument('res', action='store', help='result file, will store the lower triangular part of a distance matrix')
    p.add_argument('distance', choices=['chisq'], help='distance to use')
    p.add_argument('-r', '--rand', action='store', help='compute the distance on a random subset of specified size',
                   type=int, default=0)

    args = p.parse_args()

    X = np.load(args.data)
    n = X.shape[0]

    if n < 2:
        raise RuntimeError('Data matrix must contain at least 2 vectors (rows)')

    if args.distance == 'chisq':
        worker = worker_chisq_M
    else:
        raise RuntimeError('Unknown distance')

    if args.rand != 0:
        # subset X
        idx = np.random.random_integers(low=0, high=n, size=args.rand)
        X = X[idx, :]
        n = X.shape[0]
        np.save('random_subset.dat.npy', X)

    pdist = Parallel(n_jobs=cpu_count()) ( delayed(worker)(X[i,:], X[i+1:n,:]) for i in np.arange(0, n-1) )

    # make the list flat:
    pdist = np.array(list(itertools.chain.from_iterable(pdist)))

    # save:
    np.save(args.res, pdist)

    return
# end main()


if __name__ == '__main__':
    main()
