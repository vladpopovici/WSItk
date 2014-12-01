__author__ = 'vlad'

import numpy as np

from skimage.color import rgb2grey
from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity

from segm.tissue import tissue_components
from stain.he import rgb2he
from descriptors.txtbin import *
from descriptors.txtgrey import *
from util.intensity import requantize

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