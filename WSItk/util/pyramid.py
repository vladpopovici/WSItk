# PYRAMID - pyramidal decompositions of the images

__author__ = 'vlad'

import numpy as np
from scipy.signal import sepfir2d
from skimage.filter import gaussian_filter


def impyramid_next_level(_img, filter='burt_adelson'):
    """
    IMPYRAMID_NEXT_LEVEL: computes the next level in a Gaussian pyramidal
     decomposition.

    :param _img: numpy.ndarray
    A grey-level image (_img.ndim == 2)

    :param filter: string
    Which filter to use in removing high frequencies:
    - burt_adelson: [-0.125, 0.250, 0.375, 0.250, -0.125]
    - custom1: [0.125, 0.275, 0.375, 0.125]  (Gaussian, sigma=0.866)

    :return: numpy.ndarray
    The new level in the pyramid, an image half the size of the original.
    """

    assert (_img.ndim == 2)

    # classical low-pass filter, with the kernel from Burt & Adelson (1983):
    if filter == 'burt_adelson':
        a = 0.375
        krn = [1/4 - a/2, 1/4, a, 1/4, 1/4 - a/2]
        res = (_img - sepfir2d(_img, krn, krn))[::2, ::2]
    elif filter == 'custom1':
        res = gaussian_filter(_img, sigma=0.866)[::2, ::2]

    return res


def impyramid_next_level_rgb(_img):
    """
    IMPYRAMID_NEXT_LEVEL_RGB: computes the next level in a Gaussian
    pyramidal decomposition, for color images (RGB, or any 3-channel
    image).

    :param _img: numpy.ndarray
    A grey-level image (_img.ndim == 2)

    :return: numpy.ndarray
    The new level in the pyramid, an image half the size of the original.
    """

    res = gaussian_filter(_img, sigma=0.866, multichannel=True)

    return res[::2, ::2, :]

