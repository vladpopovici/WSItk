# -*- coding: utf-8 -*-
"""
PDIST: computes pairwise distances for large data matrices.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__version__ = 0.1
__author__ = 'Vlad Popovici'

import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
from libc.stdlib cimport abort, malloc, free

cdef extern from "strings.h":
    void bzero(void* buf, size_t n)

# distance between two vectors (histograms)
def dist_chisq(np.ndarray[np.float64_t, ndim=1] p, np.ndarray[np.float64_t, ndim=1] q):
    # Chi^2 distance between two histograms
    cdef Py_ssize_t i
    cdef double a = 0.0
    cdef double b = 0.0
    cdef double r = 0.0
    cdef size_t n = p.size

    with nogil:
        for i in range(n):
            a = (p[i] - q[i])**2
            b =  p[i] + q[i]
            if not -1e-12 < b < 1e12:
                r += a / b

    return 0.5*r

# distance between a vector and a matrix
def dist_chisq_v(np.ndarray[np.float64_t, ndim=1] p, np.ndarray[np.float64_t, ndim=2] Q,
    np.ndarray[np.float64_t, ndim=1] res):

    cdef Py_ssize_t i, n = Q.shape[0]
    cdef double* r

    r = <double *> malloc(sizeof(double) * n)
    if r == NULL:
        abort()
    bzero(r, n*sizeof(double))

    with nogil, parallel():
        for i in prange(n, schedule='guided'):
            r[i] = dist_chisq(p, Q[i,:])

    for i in xrange(n):
        res[i] = r[i]

    free(r)

    return res

