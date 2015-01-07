__author__ = 'vlad'

import numpy as np

from skimage.color import rgb2grey
from skimage.transform import rescale, resize
from skimage.util import img_as_ubyte

from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity

from segm.tissue import tissue_components
from stain.he import rgb2he
from descriptors.txtbin import *
from descriptors.txtgrey import *
from util.intensity import requantize
from util.explore import sliding_window

def extract_descriptors_he(_img, _components_models):
    w, h, nch = _img.shape
    npx = w * h
    
    n_grey_levels = 16
    glcm_window = 15
    
    # Pre-processing:
    # -get tissue basic components:
    comp_map       = tissue_components(_img, _components_models)
    img_chromatin  = (comp_map == 0).astype(np.uint8) # 0: chromatin
    img_connective = (comp_map == 1).astype(np.uint8) # 1: connective
    img_fat        = (comp_map == 2).astype(np.uint8) # 2: fat
    # -extract intensity and stains
    img_grey       = rgb2grey(_img)
    img_grey       = requantize(img_grey, nlevels=n_grey_levels, method='linear')
    img_h, img_e   = rgb2he(_img, normalize=True)
    img_h          = requantize(img_h, nlevels=n_grey_levels, method='linear')
    img_e          = requantize(img_e, nlevels=n_grey_levels, method='linear')

    gabor = GaborDescriptors()
    lbp = LBPDescriptors()
    glcm = GLCMDescriptors(glcm_window, glcm_window/3, 0, n_grey_levels)

    desc = {}
    # 1. Descriptors from binary image
    desc['bin_compact_chromatin'] = compactness(img_chromatin)
    desc['bin_compact_connective'] = compactness(img_connective)
    desc['bin_compact_fat'] = compactness(img_fat)

    desc['bin_prop_chromatin'] = np.sum(img_chromatin, dtype=np.float64) / npx
    desc['bin_prop_connective'] = np.sum(img_connective, dtype=np.float64) / npx
    desc['bin_prop_fat'] = np.sum(img_fat, dtype=np.float64) / npx

    # 2. Descriptors from grey-level image
    desc['grey_gabor'] = gabor.compute(img_grey)
    desc['grey_lbp'] = lbp.compute(img_grey)
    desc['grey_glcm'] = glcm.compute(img_grey)

    # 3. Descriptors from intensity image (Haematoxylin)
    desc['h_gabor'] = gabor.compute(img_h)
    desc['h_lbp'] = lbp.compute(img_h)
    desc['h_glcm'] = glcm.compute(img_h)

    # 4. Descriptors from intensity image (Eosin)
    desc['e_gabor'] = gabor.compute(img_e)
    desc['e_lbp'] = lbp.compute(img_e)
    desc['e_glcm'] = glcm.compute(img_e)

    return desc


def dist_descriptors_he(x1, x2, w=None):
    assert (len(x1) == len(x2))

    if w is None:
        w = np.ones((1,len(x1)))

    d = 0.0

    d += GaborDescriptors.dist(x1['grey_gabor'], x2['grey_gabor'])
    d += GaborDescriptors.dist(x1['h_gabor'], x2['h_gabor'])
    d += GaborDescriptors.dist(x1['e_gabor'], x2['e_gabor'])

    d += LBPDescriptors.dist(x1['grey_lbp'], x2['grey_lbp'])
    d += LBPDescriptors.dist(x1['h_lbp'], x2['h_lbp'])
    d += LBPDescriptors.dist(x1['e_lbp'], x2['e_lbp'])

    d += GLCMDescriptors.dist(x1['grey_glcm'], x2['grey_glcm'])
    d += GLCMDescriptors.dist(x1['h_glcm'], x2['h_glcm'])
    d += GLCMDescriptors.dist(x1['e_glcm'], x2['e_glcm'])

    return d


# get_gabor_desc
def get_gabor_desc(img, gdesc, w_size, scale=1.0, mask=None):
    """
    Extract local Gabor descriptors by scanning an image.

    :param img: numpy.ndarray
      Input intensity (grey-scale) image.

    :param gdesc: txtgrey.GaborDescriptor
      The parameters of the Gabor wavelets to be used.

    :param w_size: integer
      Window size (the sliding window is square-shaped).

    :param scale: float
      The image may be scaled prior to any descriptor extraction.

    :param mask: numpy.ndarray
      A mask (logical image) indicating the object regions
      in the image.

    :return: list
      A list with the local descriptors corresponding to each position
      of the sliding window. Each element of the list is a vector
      containing the coordinates of the local window (first 4 elements)
      and the 2 vectors of values for the local Gabor descriptors (one
      with the mean responses and one with the variances).
    """

    assert (img.ndim == 2)

    img_ = rescale(img, scale)
    if mask is not None:
        assert (mask.ndim == 2)
        assert (mask.shape == img.shape)
        mask = img_as_ubyte(resize(mask, img_.shape))

    desc = []

    img_iterator = sliding_window(img_.shape, (w_size, w_size), step=(w_size, w_size))  # non-overlapping windows

    if mask is None:
        for w_coords in img_iterator:
            z = np.hstack((np.array(w_coords),
                           gdesc.compute(img_[w_coords[0]:w_coords[1], w_coords[2]:w_coords[3]])))
            desc.append(z.tolist())
    else:
        th = w_size * w_size / 20.0   # consider only those windows with more than 5% pixels from object
        for w_coords in img_iterator:
            if mask[w_coords[0]:w_coords[1], w_coords[2]:w_coords[3]].sum() > th:
                z = np.hstack((np.array(w_coords),
                               gdesc.compute(img_[w_coords[0]:w_coords[1], w_coords[2]:w_coords[3]])))
                desc.append(z.tolist())

    return desc
# end get_gabor_desc()