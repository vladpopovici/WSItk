# -*- coding: utf-8 -*-
"""
SEGM.TISSUE: try to segment the tissue regions from a pathology slide.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

__version__ = 0.01
__author__ = 'Vlad Popovici'


import numpy as np

import skimage.morphology as skm

from sklearn.cluster import MiniBatchKMeans

import mahotas as mh

from util.intensity import _G

def tissue_region_from_rgb(_img, _min_area=150, _g_th=None):
    """
    TISSUE_REGION_FROM_RGB detects the region(s) of the image containing the
    tissue. The original image is supposed to represent a haematoxylin-eosin
    -stained pathology slide.
    
    The main purpose of this function is to detect the parts of a large image
    which most probably contain tissue material, and to discard the background.
    
    Usage:
        tissue_mask = tissue_from_rgb(img, _min_area=150, _g_th=None)
        
    Args:
        img (numpy.ndarray): the original image in RGB color space
        _min_area (int, default: 150): any object with an area smaller than 
            the indicated value, will be discarded
        _g_th (int, default: None): the processing is done on the GREEN channel
            and all pixels below _g_th are considered candidates for "tissue
            pixels". If no value is given to _g_th, one is computed by K-Means
            clustering (K=2), and is returned.
        
    Returns:
        numpy.ndarray: a binary image containing the mask of the regions
            considered to represent tissue fragments
        int: threshold used for GREEN channel
    """
    
    if _g_th is None:
    # Apply vector quantization to remove the "white" background - work in the
    # green channel:
        vq = MiniBatchKMeans(n_clusters=2)
        _g_th = int(np.round(0.95 * np.max(vq.fit(_G(_img).reshape((-1,1)))
            .cluster_centers_.squeeze())))
    
    mask = _G(_img) < _g_th

    skm.binary_closing(mask, skm.disk(3), out=mask)
    
    skm.remove_small_objects(mask, min_size=_min_area, in_place=True)
    mask = mh.close_holes(mask)
    
    return mask, _g_th