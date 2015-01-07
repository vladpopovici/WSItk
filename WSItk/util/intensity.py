"""
INTENSITY: utility functions for intensity transformation.

@author: vlad
"""
__version__ = 0.01
from cv2 import kmeans
#from atk import Image

__author__ = 'Vlad Popovici'
__all__ = ['invert', '_R', '_G', '_B', 'requantize']

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from skimage.util import dtype_limits
from skimage.exposure import rescale_intensity


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


def requantize(img, nlevels=2, method='linear'):
    """
    REQUANTIZE: changes the number of grey scale levels in an image.
    
    Usage:
        res = requantize(img, nlevels=2)
    
    Args:
        img (numpy.ndarray): a single channel image
        nlevels (int): number of levels in the resulting image (<256)
        method (string): 'linear' or 'adaptive'
            'linear': the interval 0..max(dtype) is split into nlevels
            equal length intervals onto which input values are mapped
            'kmeans': vector quantization is applied to determine
            the new quantification levels
            
    Returns:
        numpy.ndarray: a single image of type uint8
        
    Raises:
        ValueError: if the image is not single channel, or if invalid
        arguments are given
    """
    
    if img.ndim != 2:
        raise ValueError('A single channel image is expected')
    
    assert(1 < nlevels <= 256 )
    assert(method.lower() in ['linear', 'kmeans'])
    
    if method.lower() == 'linear':
        res = np.ndarray(img.shape, dtype=np.uint8)
        res.fill(nlevels - 1)                        # nlevels-1 is the highest grey level
        limits = np.linspace(0, 256, nlevels+1, dtype=np.uint8)
        img = rescale_intensity(img, out_range=(0, 255))
        for k in np.arange(start=nlevels-2, stop=-1, step=-1):
            res[img < limits[k]] = k
    elif method.lower() == 'kmeans':
        vq = MiniBatchKMeans(n_clusters=nlevels)
        vq.fit(img.reshape((-1,1)))
        res = vq.labels_.reshape(img.shape).astype(np.uint8)
    else:
        ValueError('Incorrect method specified')
             
    return res

