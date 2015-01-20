# UTIL.MATH: small math functions

__author__ = 'vlad'

import numpy as np
from numpy import dot

from scipy.stats import entropy
from scipy.linalg import norm

class dist:
    @staticmethod
    def cosine(x_, y_):
        return dot(x_, y_) / (norm(x_)*norm(y_))


    @staticmethod
    def euclid(x_, y_):
        return norm(x_, y_)


    @staticmethod
    def kl(x_, y_):
        # Kullback-Leibler
        return 0.5*(entropy(x_, y_) + entropy(y_, x_))


    @staticmethod
    def js(x_, y_):
        # Jensen-Shannon
        return 0.5*(entropy(x_, 0.5*(x_+y_))+entropy(y_,0.5*(x_+y_)))


    @staticmethod
    def bhattacharyya(x_, y_):
        # Bhattacharyya distance between histograms
        return -np.log(np.sum(np.sqrt(x_*y_)))


    @staticmethod
    def matusita(x_, y_):
        # Matusita distance between histograms
        return np.sqrt(np.sum((np.sqrt(x_)-np.sqrt(y_))**2))

# end class dist
