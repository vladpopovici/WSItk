"""
EXTRACT: extracts a variety of local descriptors
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)

__author__ = 'vlad'
__version__ = 0.2
__all__ = ['get_gabor_desc', 'dist_gabor_desc', 'pdist_gabor', 
           'extract_descriptors_he', 'pairwise_distances',
           'get_local_desc', 'desc_to_matrix', 'pdist_hog', 'dist_hog_desc',
           'pdist_lbp', 'dist_lbp_desc', 'pdist_mfs', 'dist_mfs_desc', 'matrix_to_desc']


from concurrent import futures
import time

import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist

from concurrent.futures import ProcessPoolExecutor, as_completed

from skimage.color import rgb2grey
from skimage.transform import rescale, resize
from skimage.util import img_as_ubyte

from skimage.util import img_as_ubyte
from skimage.exposure import rescale_intensity
from skimage.color import rgb2hsv

from segm.tissue import tissue_components
from stain.he import rgb2he, rgb2he2
from descriptors.txtbin import *
from descriptors.txtgrey import *
from util.intensity import requantize
from util.explore import sliding_window
from util.math import dist

# local worker
def _gabor_worker(_img, _feat, _box):
    return {'roi': _box, 'gabor': _feat.compute(_img[_box[0]:_box[1], _box[2]:_box[3]])}


def dist_gabor_desc(d1, d2, _method="euclidean"):
    return GaborDescriptor.dist(d1['gabor'], d2['gabor'], _method)


# get_gabor_desc
def get_gabor_desc(img, gdesc, w_size, scale=1.0, mask=None, _ncpus=None):
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

    img_iterator = sliding_window(img_.shape, (w_size, w_size), step=(w_size, w_size))  # non-overlapping windows

    res = []
    if mask is None:
        with ProcessPoolExecutor(max_workers=_ncpus) as executor:
            for w_coords in img_iterator:
                time.sleep(0.01)
                res.append(executor.submit(_gabor_worker, img_, gdesc, w_coords))
    else:
        th = w_size * w_size / 20.0   # consider only those windows with more than 5% pixels from object
        with ProcessPoolExecutor(max_workers=_ncpus) as executor:
            for w_coords in img_iterator:
                time.sleep(0.01)
                if mask[w_coords[0]:w_coords[1], w_coords[2]:w_coords[3]].sum() > th:
                    res.append(executor.submit(_gabor_worker, img_, gdesc, w_coords))

    desc = []
    for f in as_completed(res):
        desc.append(f.result())

    return desc
# end get_gabor_desc()


def _worker2(_hue, _h, _e, _gabor, _lbp, _roi):
    """
    Computes the local descriptors on a patch.
    :param _hue:
    :param _h:
    :param _e:
    :param _gabor:
    :param _lbp:
    :param _roi:
    :return:
    """
    hue_mean = _hue[_roi[0]:_roi[1], _roi[2]:_roi[3]].mean()
    hue_std  = _hue[_roi[0]:_roi[1], _roi[2]:_roi[3]].std()

    g_h = _gabor.compute(_h[_roi[0]:_roi[1], _roi[2]:_roi[3]])
    g_e = _gabor.compute(_e[_roi[0]:_roi[1], _roi[2]:_roi[3]])

    l_h = _lbp.compute(_h[_roi[0]:_roi[1], _roi[2]:_roi[3]])
    l_e = _lbp.compute(_e[_roi[0]:_roi[1], _roi[2]:_roi[3]])

    res = {'roi': _roi, 'hue_mean': hue_mean, 'hue_std': hue_std,
           'gabor_haem': g_h, 'gabor_eos': g_e, 'lbp_haem': l_h, 'lbp_eos': l_e}

    return res


# EXTRACT_LOCAL_DESCRIPTORS_HE
def extract_descriptors_he(_img, w_size, _ncpus=None):
    """
    EXRACT_LOCAL_DESCRIPTORS_HE: extracts a set of local descriptors of the image:
        - histogram of Hue values
        - histogram of haematoxylin and eosin planes
        - Gabor descriptors in haematoxylin and eosin spaces, respectively
        - local binary patterns in haematoxylin and eosin spaces, respectively

    :param _img: numpy.ndarray

    :param w_size: int

    :return: list
    """
    assert (_img.ndim == 3)

    img_iterator = sliding_window(_img.shape[:-1], (w_size, w_size), step=(w_size, w_size))  # non-overlapping windows
    gabor = GaborDescriptor()
    lbp   = LBPDescriptor()

    hsv = rgb2hsv(_img)
    h, e, _ = rgb2he2(_img)

    res = []
    with ProcessPoolExecutor(max_workers=_ncpus) as executor:
        for w_coords in img_iterator:
            res.append(executor.submit(_worker2, hsv[:,:,0], h, e, gabor, lbp, w_coords))

    desc = []
    for f in as_completed(res):
        desc.append(f.result())

    return desc


def _dist(d1, d2):
    d = 0.0
    d += GaborDescriptor.dist(d1['gabor_haem'], d2['gabor_haem'])
    d += GaborDescriptor.dist(d1['gabor_eos'], d2['gabor_eos'])
    d += LBPDescriptor.dist(d1['lbp_haem'], d2['lbp_haem'], method='bh')
    d += LBPDescriptor.dist(d1['lbp_eos'], d2['lbp_eos'], method='bh')
    d += (d1['hue_mean'] - d2['hue_mean']) / (np.sqrt(0.5*(d1['hue_std']**2) + d2['hue_std']**2))   # t-stats like

    return d


# pairwise distances

def pairwise_distances(desc, _distfn=_dist, set_to_zero=1e-16):
    """
    PAIRWISE_DISTANCES: computes the pairwise distances between local descriptors.

    :param desc: list
     A list of descriptors, as returned by get_*_desc() functions
    :return: numpy.ndarray
     The lower triangle of the square matrix with distances. (As returned by
     pdist() function from scipy.spatial.distance.)
    """
    n = len(desc)
    d = np.zeros((n*(n-1)/2))
    for i in np.arange(1, n):
        for j in np.arange(i):
            d[n*j - j*(j+1)/2 + i - 1 - j] = _distfn(desc[i], desc[j])
    d[d <= set_to_zero] = 0.0

    return d
# end pairwise_distances


def _dist_worker_gabor(list_ft, i):
    # Compute the distance between list_ft[i] and each element in list_ft[idx]
    d = map(dist_gabor_desc, list_ft[:i], [list_ft[i]]*i)
    d.insert(0, i)
    
    return d
    

def pdist_gabor(desc, set_to_zero=1e-16, _ncpus=None):
    n = len(desc)
    l = []
    with ProcessPoolExecutor(max_workers=_ncpus) as executor:
        for i in np.arange(1, n):
            time.sleep(0.01)
            l.append(executor.submit(_dist_worker_gabor, desc, i))

    d = np.zeros((n*(n-1)/2))            
    for x in as_completed(l):
        res = x.result()
        i = res[0]
        j = np.arange(i)
        idx = (n*j - j*(j+1)/2 + i - 1 - j).astype(np.int64)
        d[idx] = res[1:]
        
    d[d <= set_to_zero] = 0.0

    return d


def dist_lbp_desc(d1, d2, _method="bh"):
    return LBPDescriptor.dist(d1['lbp'], d2['lbp'], _method)


def _dist_worker_lbp(list_ft, i):
    # Compute the distance between list_ft[i] and each element in list_ft[idx]
    d = map(dist_lbp_desc, list_ft[:i], [list_ft[i]]*i)
    d.insert(0, i)

    return d


def pdist_lbp(desc, set_to_zero=1e-16, _ncpus=None):
    n = len(desc)
    l = []
    with ProcessPoolExecutor(max_workers=_ncpus) as executor:
        for i in np.arange(1, n):
            time.sleep(0.01)
            l.append(executor.submit(_dist_worker_lbp, desc, i))

    d = np.zeros((n*(n-1)/2))
    for x in as_completed(l):
        res = x.result()
        i = res[0]
        j = np.arange(i)
        idx = (n*j - j*(j+1)/2 + i - 1 - j).astype(np.int64)
        d[idx] = res[1:]

    d[d <= set_to_zero] = 0.0

    return d


def dist_hog_desc(d1, d2, _method="bh"):
    return HOGDescriptor.dist(d1['hog'], d2['hog'], _method)


def _dist_worker_hog(list_ft, i):
    # Compute the distance between list_ft[i] and each element in list_ft[idx]
    d = map(dist_hog_desc, list_ft[:i], [list_ft[i]]*i)
    d.insert(0, i)

    return d


def pdist_hog(desc, set_to_zero=1e-16, _ncpus=None):
    n = len(desc)
    l = []
    with ProcessPoolExecutor(max_workers=_ncpus) as executor:
        for i in np.arange(1, n):
            time.sleep(0.01)
            l.append(executor.submit(_dist_worker_hog, desc, i))

    d = np.zeros((n*(n-1)/2))
    for x in as_completed(l):
        res = x.result()
        i = res[0]
        j = np.arange(i)
        idx = (n*j - j*(j+1)/2 + i - 1 - j).astype(np.int64)
        d[idx] = res[1:]

    d[d <= set_to_zero] = 0.0

    return d


def dist_mfs_desc(d1, d2, _method="euclidean"):
    return MFSDescriptor.dist(d1['mfs'], d2['mfs'], _method)


def _dist_worker_mfs(list_ft, i):
    # Compute the distance between list_ft[i] and each element in list_ft[idx]
    d = map(dist_mfs_desc, list_ft[:i], [list_ft[i]]*i)
    d.insert(0, i)

    return d


def pdist_mfs(desc, set_to_zero=1e-16, _ncpus=None):
    n = len(desc)
    l = []
    with ProcessPoolExecutor(max_workers=_ncpus) as executor:
        for i in np.arange(1, n):
            time.sleep(0.01)
            l.append(executor.submit(_dist_worker_mfs, desc, i))

    d = np.zeros((n*(n-1)/2))
    for x in as_completed(l):
        res = x.result()
        i = res[0]
        j = np.arange(i)
        idx = (n*j - j*(j+1)/2 + i - 1 - j).astype(np.int64)
        d[idx] = res[1:]

    d[d <= set_to_zero] = 0.0

    return d

# local worker
def _desc_worker(_img, _feat, _box, _label):
    return {'roi': _box, _label: _feat.compute(_img[_box[0]:_box[1], _box[2]:_box[3]])}


# get_local_desc
def get_local_desc(img, desc, img_iterator, label, _ncpus=None):
    """
    Extract local descriptors by scanning an image, according to a pre-specified
    algorithm.

    :param img: numpy.ndarray
      Input intensity (grey-scale) image.

    :param desc: txtgrey.LocalDescriptors
      An object encoding the local descriptor.

    :param img_iterator: iterator
      An iterator over the image. See util.explore for various iterators.

    :param label: string
      A label to identify the descriptor.

    :return: list
      A list with the local descriptors corresponding to each position
      of the sliding window. Each element of the list is a vector
      containing the coordinates of the local window (first 4 elements)
      and the result of applying the local descriptor over image[roi].
    """

    assert (img.ndim == 2)

    res = []
    with ProcessPoolExecutor(max_workers=_ncpus) as executor:
        for w_coords in img_iterator:
            time.sleep(0.01)
            res.append(executor.submit(_desc_worker, img, desc, w_coords, label))

    desc = []
    for f in as_completed(res):
        desc.append(f.result())

    return desc
# end get_local_desc()


def desc_to_matrix(desc, label):
    """
    Converts a list of descriptors to a matrix representation, with one
    sample per row. First 4 elements of each row contain the ROI coordinates.

    :param desc:
       list of descriptors, as returned by get_*_desc() functions
    :param label:
       the label of the descriptors to be extracted from the list
    :return: numpy.ndarray
    """

    l = [d[label] for d in desc]
    r = [d['roi'] for d in desc]

    if len(l) == 0:
        # no such descriptors
        return None

    return np.hstack( (np.array(r), np.array(l)) )


def matrix_to_desc(mtx, label):
    """
    Converts a matrix to a list of descriptors. It is supposed that the input matrix
    was produced by (or conforms to the format output by) desc_to_matrix() function.
    
    :param mtx:
        A numpy.ndarray with one descriptor per row.
        
    :param label:
        The label of these descriptors.
        
    :return:
        A list of descriptors.
    """

    desc = []
    for i in np.arange(mtx.shape[0]):
        desc.append({'roi': np.array(mtx[i,0:4], dtype=np.int), label: mtx[i,5:]})

    return desc
