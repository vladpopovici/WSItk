## This code was adapted from the code at
## https://www.mail-archive.com/scikit-learn-general@lists.sourceforge.net/msg12635.html

from __future__ import division
from builtins import *

import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

def gap(X, Ks, Wstar=None, B=0):
    """
    Gap statistic for estimating the "optimal" number of clusters.
    For details see [1]_.
    
    :param X: numpy.ndarray
        Input data, as a matrix with samples by rows.
    :param Ks: enumerable
        A list or numpy.vector or different numbers of clusters
        to be tried.
    :param Wstar: numpy.array
        Estimates of the distribution of W* under a uniform random data
        distribution. Can be reused in case of repeated estimations of K
        for different (but similar) data sets X. If None, a new W* will
        be generated.
    :param B: int
        Number of random data sets to be used in estimating
        the null distribution. If Wstar is not None, B is obtained as
        the number of rows of Wstar.
    :return: a tuple
        Two elements are returned:
        -the estimated "optimal" K
        -the matrix Wstar (see above)
        
    References:
    -----------
    .. [1] Tibshirani, Walther, Hastie. Estimating the number of clusters
       in a data set via the gap statistic. J. R. Statist. Soc. B (2001) 63,
       Part 2, pp 411-423. See: http://web.stanford.edu/~hastie/Papers/gap.pdf
    """

    #Get the bounding box of the population

    mins, maxes = np.min(X, axis = 0), np.max(X, axis = 0)
    Ks = np.array(Ks)
    Ks.sort()                                    # make sure they are sorted increasingly
    nk = Ks.size
    
    K_max = Ks.max()                             # maximum number of clusters
    
    mbk = MiniBatchKMeans(compute_labels=True)
    
    if Wstar is None:
        # generate the null distribution
        Wstar = np.zeros([B, nk])                # dispersion of random dataset b with k+1 clusters
        
        for b in range(B):
            dataset = np.random.rand(*X.shape) * (maxes-mins) + mins
            for i, k in enumerate(Ks):
                mbk.n_clusters = k+1
                mbk.fit(dataset)
                Wstar[b, i] = np.log(mbk.inertia_)

    W = np.zeros(nk)                         # dispersion
    
    for i, k in enumerate(Ks):
        mbk.n_clusters = k+1 
        mbk.fit(X)
        W[i] = np.log(mbk.inertia_)
            
    Gap = Wstar.mean(axis=0) - W
    s = np.std(Wstar, axis=0) / np.sqrt(1.0 + 1.0/B)
    kl = [k for i, k in enumerate(Ks[:-1]) if Gap[i] >= Gap[i+1] - s[i+1]]
    if len(kl) > 0:
        # at least one reasonable k:
        khat = min(kl)
    else:
        # just resturn the max:
        khat = Ks[-1]
    
    return khat, Wstar

