"""
STAIN.NORM: various methods for stain normalization.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

__author__ = 'vlad'
__version__ = 0.1

import numpy as np

from skimage.util import img_as_float
from skimage.exposure import rescale_intensity

def compute_macenko_norm_matrix(im, alpha=1.0, beta=0.15):
    """
    Implements the staining normalization method from
      Macenko M. et al. "A method for normalizing histology slides for
      quantitative analysis". ISBI 2009

    :param im:
    :param alpha:
    :param beta:

    :return:
    """
    if im.ndim != 3:
        raise ValueError('Input image must be RGB')
    h, w, _ = im.shape

    im = (im + 1.0) / 255.0 # img_as_float(im)
    # im = rescale_intensity(im, out_range=(0.001, 1.0))  # we'll take log...
    im = im.reshape((h*w, 3), order='F')
    od = -np.log(im)                 # optical density
    odhat = od[~np.any(od < beta, axis=1), ]
    _, V = np.linalg.eigh(np.cov(odhat, rowvar=0))  # eigenvectors of a symmetric matrix
    theta = np.dot(odhat,V[:, 1:3])
    phi = np.arctan2(theta[:,1], theta[:,0])
    minPhi, maxPhi = np.percentile(phi, [alpha, 100-alpha])
    vec1 = np.dot(V[:,1:3] , np.array([[np.cos(minPhi)],[np.sin(minPhi)]]))
    vec2 = np.dot(V[:,1:3] , np.array([[np.cos(maxPhi)],[np.sin(maxPhi)]]))
    stain_matrix = np.zeros((3,3))
    if vec1[0] > vec2[0]:
        stain_matrix[:, :2] = np.hstack((vec1, vec2))
    else:
        stain_matrix[:, :2] = np.hstack((vec2, vec1))

    stain_matrix[:, 2] = np.cross(stain_matrix[:, 0], stain_matrix[:, 1])

    # he1_from_rgb = linalg.inv(rgb_from_he1)
    return stain_matrix.transpose()
