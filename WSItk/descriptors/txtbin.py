"""
TXTBIN: binary textural descriptors. Mostly, wrappers for functions from various
other packages.
"""

import numpy as np

from skimage.filters.rank import median
from skimage.morphology import disk


def compactness(_img):

    ## Lookup table for the empirical distribution of median(img)/img
    ## in the case of a random image (white noise), for varying
    ## proportions of white pixels in the image (0.1, 0.2, ..., 1.0).
    ## The distributions are approximated by Gaussians, and the
    ## corresponding means and standard deviations are stored.

    prop = np.linspace(0.1, 1.0, 10)
    emp_distrib_mean = np.array([4.4216484763437432e-06, 0.0011018116582350559,
                                 0.042247116747488218, 0.34893587605251208,
                                 1.0046008733628913, 1.4397675817057451,
                                 1.4115741958770296, 1.2497935146232551,
                                 1.1111058415275834, 1.0])
    emp_distrib_std = np.array([2.7360459073474441e-05, 0.00051125394394966434,
                                0.0038856377648490894, 0.012029872915543046,
                                0.013957075037020938, 0.0057246251730834283,
                                0.0028750796874699143, 0.0023709207886137384,
                                0.0015018959493632007, 0.0])
    if _img.ndim != 2:
        raise ValueError('The input image must be a 2D binary image.')

    _img = (_img != 0).astype(np.uint8)
    _med = median(_img, disk(3))

    # "proportion of white pixels" in the image:
    swp = np.sum(_img, dtype=np.float64)
    pwp = swp / _img.size

    # compactness coefficient
    cf = np.sum(_med, dtype=np.float64) / swp

    # standardize using the "closest" Gaussian from the list of empirical
    # distributions:
    k = np.argmin(np.abs(prop - pwp))

    # this should make the coeff more or less normally distributed N(0,1)
    cf = (cf - emp_distrib_mean[k]) / emp_distrib_std[k]

    return cf