"""
VISUALIZATION: several functions for visualizing the results of various image analysis procedures.
"""
__author__ = 'vlad'
__version__ = 0.1
__all__ = ['enhance_patches']


from skimage.exposure import adjust_gamma


def enhance_patches(_img, _patches, _gamma=0.2):
    """
    Emphasize a set of rectangular regions from an image. The original image is altered such that
    the regions of interest appear enhanced.

    :param _img: numpy.ndarray
       original image
    :param _patches: list
       list of 4-elements vectors indicating the coordinates of patches of interest
    :param _gamma: float
       gamma adjustment for "background" image
    :return:
    """

    _res = adjust_gamma(_img, _gamma)

    for p in _patches:
        _res[p[0]:p[1], p[2]:p[3]] = _img[p[0]:p[1], p[2]:p[3]]

    return _res
