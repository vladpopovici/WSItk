"""
HE module: responsible for color processing for images of H&E-stained tissue samples.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)


__version__ = 0.01
__author__ = 'Vlad Popovici'
__all__ = ['rgb2he']

import numpy as np
from scipy import linalg
from skimage.color import separate_stains


def rgb2he(img):
    """
    RGB2HE: Extracts the haematoxylin and eosin components from an RGB color.

    h,e = rgb2he(img)

    Args:
        img (numpy.ndarray): and RGB image; no check for actual color space is
        performed

    Returns:
        tuple. Contains two intensity images as numpy.ndarray, coding for the
        haematoxylin and eosin components, respectively
    """

    # my color separation matrices
    # (from http://www.mecourse.com/landinig/software/colour_deconvolution.zip)

    rgb_from_he1 = np.array([[0.644211, 0.716556, 0.266844],
                             [0.092789, 0.954111, 0.283111],
                             [0.0, 0.0, 0.0]])
    rgb_from_he1[2, :] = np.cross(rgb_from_he1[0, :], rgb_from_he1[1, :])
    he1_from_rgb = linalg.inv(rgb_from_he1)

    img_tmp = separate_stains(img, he1_from_rgb)

    return img_tmp[:, :, 0], img_tmp[:, :, 1]
