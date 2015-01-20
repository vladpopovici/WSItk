from __future__ import (absolute_import, division, print_function, unicode_literals)

__author__ = 'vlad'
__version__ = 0.2

import numpy as np
from scipy import ndimage

# from sklearn.cluster import KMeans

import skimage.morphology as morph
# from skimage.restoration import denoise_tv_bregman
from skimage.feature import peak_local_max

import mahotas

# from stain.he import rgb2he


HE_OPTS = {'gauss1': np.sqrt(2.0),
           'gauss2': 1.0/np.sqrt(2.0),
           'strel1':  morph.disk(3),
           'bregm': 3.5,
           # options for nuclei extraction at 40x magnification:
           '40x_nuclei_min_area': 30}


# NUCLEI_REGIONS
def nuclei_regions(comp_map):
    """
    NUCLEI_REGIONS: extract "support regions" for nuclei. This function
    expects as input a "tissue components map" (as returned, for example,
    by segm.tissue_components) where values of 1 indicate pixels having
    a color corresponding to nuclei.
    It returns a set of compact support regions corresponding to the
    nuclei.


    :param comp_map: numpy.ndarray
       A mask identifying different tissue components, as obtained
       by classification in RGB space. The value 0

       See segm.tissue.tissue_components()

    :return:
    """
    # Deprecated:...
    # img_hem, _ = rgb2he(img0, normalize=True)

    # img_hem = denoise_tv_bregman(img_hem, HE_OPTS['bregm'])

    # Get a mask of nuclei regions by unsupervised clustering:
    # Vector Quantization: background, mid-intensity Hem and high intensity Hem
    # -train the quantizer for 3 levels
    # vq = KMeans(n_clusters=3)
    # vq.fit(img_hem.reshape((-1,1)))
    # -the level of interest is the brightest:
    # k = np.argsort(vq.cluster_centers_.squeeze())[2]
    # mask_hem = (vq.labels_ == k).reshape(img_hem.shape)
    # ...end deprecated

    # Final mask:
    mask = (comp_map == 1)   # use the components classified by color

    # mask = morph.closing(mask, selem=HE_OPTS['strel1'])
    # mask = morph.opening(mask, selem=HE_OPTS['strel1'])
    # morph.remove_small_objects(mask, in_place=True)
    # mask = (mask > 0)

    mask = mahotas.close_holes(mask)
    morph.remove_small_objects(mask, in_place=True)

    dst  = mahotas.stretch(mahotas.distance(mask))
    Bc=np.ones((9,9))
    lmax = mahotas.regmax(dst, Bc=Bc)
    spots, _ = mahotas.label(lmax, Bc=Bc)
    regions = mahotas.cwatershed(lmax.max() - lmax, spots) * mask

    return regions
# end NUCLEI_REGIONS