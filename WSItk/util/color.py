"""
COLOR: a few functions to complement the COLOR module in scikit-image package.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

__version__ = 0.01
__author__ = 'Vlad Popovici'

import numpy as np
from skimage.util import img_as_uint

def rgb2ycbcr(im):
    """
    RGB2YCBCR: converts an RGB image into YCbCr (YUV) color space.
    
    :param im: numpy.ndarray
      [m x n x 3] image
    """
    
    if im.ndim != 3:
        raise ValueError('Input image must be RGB.')
    h, w, c = im.shape
    if c != 3:
        raise ValueError('Input image must be a 3-channel (RGB) image.')
    
    if im.dtype != np.uint8:
        im = img_as_uint(im)
    
    ycc = np.array([[0.257,  0.439, -0.148],
                    [0.504, -0.368, -0.291],
                    [0.098, -0.071,  0.439]])
    
    im = im.reshape((h*w, c))
    
    r = np.dot(im, ycc).reshape((h, w, c))
    r[:,:,0] += 16
    r[:,:,1:3] += 128
    
    im_res = np.array(np.round(r), dtype=im.dtype)
    
    return im_res
    

def ycbcr2rgb(im):
    """
    YCBCR2RGB: converts an YCbCr (YUV) in RGB color space.
    
    :param im: numpy.ndarray
      [m x n x 3] image
    """

    if im.ndim != 3:
        raise ValueError('Input image must be YCbCr.')
    h, w, c = im.shape
    if c != 3:
        raise ValueError('Input image must be a 3-channel (YCbCr) image.')
    
    if im.dtype != np.uint8:
        im = img_as_uint(im)

    iycc = np.array([[1.164,  1.164,  1.164],
                     [0,     -0.391,  2.018],
                     [1.596, -0.813,  0]])
    
    r = im.reshape((h*w, c))    
    
    r[:, 0] -= 16.0
    r[:, 1:3] -= 128.0
    r = np.dot(r, iycc)
    r[r < 0] = 0
    r[r > 255] = 255
    r = np.round(r)
    #x = r[:,2]; r[:,2] = r[:,0]; r[:,0] = x

    im_res = np.array(r.reshape((h, w, c)), dtype=np.uint8)
    
    return im_res
