"""
HE module: responsible for color processing for images of H&E-stained tissue samples.
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)


__version__ = 0.01
__author__ = 'Vlad Popovici'
__all__ = ['rgb2he']

import numpy as np
#from scipy import linalg
from skimage.color import separate_stains
from skimage import img_as_ubyte

def rgb2he(img, normalize=False):
    """
    RGB2HE: Extracts the haematoxylin and eosin components from an RGB color.

    h,e = rgb2he(img, normalize=False)

    Args:
        img (numpy.ndarray): and RGB image; no check for actual color space is
        performed
        
        normalize (bool): should the values be linearly transformed, to ensure
        they are all between -1.0 and 1.0  

    Returns:
        tuple. Contains two intensity images as numpy.ndarray, coding for the
        haematoxylin and eosin components, respectively. The values are in the
        "H&E" space or in -1..1 (if normalize==True)
    """

    # my color separation matrices
    # (from http://www.mecourse.com/landinig/software/colour_deconvolution.zip)
    # The code commented out below was used to generate the "he1_from_rgb" matrix.
    # After the matrix was obtained, it has been hard coded, for computation 
    # efficiency:
    #rgb_from_he1 = np.array([[0.644211, 0.716556, 0.266844],
    #                         [0.092789, 0.954111, 0.283111],
    #                         [0.0, 0.0, 0.0]])
    #rgb_from_he1[2, :] = np.cross(rgb_from_he1[0, :], rgb_from_he1[1, :])
    #he1_from_rgb = linalg.inv(rgb_from_he1)
    he1_from_rgb = np.array([[ 1.73057512, -1.3257525 , -0.1577248 ],
                             [-0.19972397,  1.1187028 , -0.48055639],
                             [ 0.10589662,  0.19656106,  1.67121469]])

    img_tmp = separate_stains(img_as_ubyte(img), he1_from_rgb)
    
    ## The RGB -> H&E transformation maps:
    ## black (0,0,0)       |-> (-1.13450710, 0.00727017021)
    ## white (255,255,255) |-> (-9.08243792, 0.05.82022531)
    ## red   (255,0,0)     |-> (-9.53805685, 6.44503007)
    ## green (0,255,0)     |-> (-0.164661728, -5.42507111)
    ## blue  (0,0,255)     |-> (-1.64873355, -0.947216369)
    
    if normalize:
        img_tmp[:,:,0] = 2*(img_tmp[:,:,0] + 9.53805685) / (-0.164661728 + 9.53805685) - 1
        img_tmp[img_tmp[:,:,0] < -1.0, 0] = -1.0
        img_tmp[img_tmp[:,:,0] >  1.0, 0] =  1.0
        
        img_tmp[:,:,1] = 2*(img_tmp[:,:,1] + 5.42507111) / (6.44503007 + 5.42507111) - 1
        img_tmp[img_tmp[:,:,1] < -1.0, 0] = -1.0
        img_tmp[img_tmp[:,:,1] >  1.0, 0] =  1.0
        
    return img_tmp[:, :, 0], img_tmp[:, :, 1]
