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

    @staticmethod
    def chisq(x_, y_):
        # Chi^2 distance between two histograms
        a = (x_ - y_)**2
        b = x_ + y_
        # to avoid diving by 0
        i = np.isclose(b, 1e-16)
        if i.any():
            a[i] = 0
            b[i] = 1

        return 0.5*np.sum(a / b)

# end class dist
