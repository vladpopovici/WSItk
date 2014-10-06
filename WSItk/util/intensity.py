# -*- coding: utf-8 -*-
"""
UTIL: utility functions for WSItk.

@author: vlad
"""
__version__ = 0.01
__author__ = 'Vlad Popovici'
__all__ = ['invert', '_R', '_G', '_B']

import numpy as np

from skimage.util import dtype_limits

def invert(img, mx=None):
    """
    INVERT applies inverse video transformation on single channel images.
    
    Usage:
        res = invert(img, max_intensity)
        
    Args:
        img (numpy.ndarray): single channel image
        max_intensity (implicit none): the maximum intensity of the type of
            images provided as input (e.g. a 8 bit/pixel image has normally
            max_intensity=255)
        
    Returns:
        numpy.ndarray: an image of the same shape as the original, but with
        the intensity levels inverted (max_intensity - initial_value)
        
    Raises:
        ValueError: if the input image is not single channel
    """
    
    if img.ndim != 2:
        raise ValueError('A single channel image is expected')
        
    if mx is None:
        mx = dtype_limits(img)[1]
    else:
        mx = np.min(dtype_limits(img)[1], mx)
        
    res = mx - img
    
    return res
    

###
# Aliases for color channels:    
def _R(_img):
    return _img[:,:,0]
    
def _G(_img):
    return _img[:,:,1]
    
def _B(_img):
    return _img[:,:,2]
