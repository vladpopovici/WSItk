# https://www.mail-archive.com/scikit-learn-general@lists.sourceforge.net/msg12635.html
from __future__ import division
import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans

def gap(X):
    '''
    See http://web.stanford.edu/~hastie/Papers/gap.pdf
    X has shape (n_samples, n_features) 
    '''
    #Get the bounding box of the population
    mins, maxes = np.min(X, axis = 0), np.max(X, axis = 0)
    B = 10        # number of reference datasets (from uniform distribution)
    K = 9         # maximum number of clusters
    W = np.zeros(K)                      # dispersion of im with k+1 clusters
    Wstar = np.zeros([B, K])         # dispersion of dataset b with k+1 clusters
    mbk = MiniBatchKMeans()
    for k in range(K):
        mbk.n_clusters = k+1 
        mbk.fit(X)
        W[k] = np.log(mbk.inertia_)
    
    for b in range(B):
        dataset = np.random.rand(*X.shape) * (maxes-mins) + mins
        for k in range(K):
            mbk.n_clusters = k+1
            mbk.fit(dataset)
            Wstar[b, k] = np.log(mbk.inertia_)
            
    Gap = Wstar.mean(axis=0) - W
    s = np.std(Wstar, axis=0) / np.sqrt(1+1/B)
    try:
        khat = 1+ min( [k for k in range (K-1) if Gap[k] >= Gap[k+1] - s[k+1]])
    except ValueError:
        khat = None
    z = np.array([0])
    Gap = np.hstack((z, Gap))
    s = np.hstack((z, s))
    return khat, Gap, s

if __name__ == '__main__':
    from sklearn import datasets
    iris = datasets.load_iris()
    X = iris.data[:, :2]  # we only take the first two features.
    k, g, s = gap(X)
    print(k)
    print(g)
    print(s)
