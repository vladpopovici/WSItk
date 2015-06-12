"""
MISC: various functions that did not fit anywhere else.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

__author__ = 'vlad'


import numpy as np
from future.utils import bytes_to_native_str as nstr
from skimage.transform import integral_image

def intg_image(image):
    """
    Computes tha integral image following the convention that the
    first row and column should be 0s.

    :param image: numpy.array
     A 2D array (single channel image).

    :return:
     A 2D array, with shape (image.shape[0]+1, image.shape[1]+1).
    """

    if image.ndim != 2:
        raise ValueError('The image must be single channel.')

    image = np.pad(integral_image(image), ((1,0),(1,0)), mode=nstr(b'constant'), constant_values=0)

    return image
# end intg_image