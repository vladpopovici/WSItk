"""
SEGM.BASIC: additional (to SciKitImage) functions for segmentation.
"""
__version__ = 0.01
__author__ = 'Vlad Popovici'
__all__ = ['bounding_box', 'connectivity', 'nuclei_region_merge']

import numpy as np

import skimage as ski
from skimage import segmentation
from skimage import filter


# BOUNDING_BOX
def bounding_box(image, th=0):
    """
    BOUNDING_BOX: extracts the coordinates of the bounding box of an object.

    res = bounding_box(image)

    Parameters
    ----------
    image: a 2D image (ndarray): everything above <th> is considered foreground.
        No test for connectiveness is performed.

    th: integer
        The threshold above which a pixel is considered to belong to an
        object. Defaults to 0.

    Returns
    -------
    res: (r0, c0, r1, c1)
        the coordinates upper-left and lower-right of the bounding box
    """

    (r,c) = np.where(image > th)

    return min(r), min(c), max(r), max(c)


# CONNECTIVITY
def connectivity(labels, neigh=None):
    """
    CONNECTIVITY: given a labeling of an image, find the neighbors for each label.

    cmap = connectivity(labels, neigh=None, keep_bakground)

    Parameters
    ----------
    labels: [M x N, uint64] a pseudo-image with a labeling of objects; label==0
        indicates background
    neighborhood: type of neighborhood: either a neighborhood matrix or None
        If None, a 4-connected neighborhood is assumed,
                     | 0 1 0 |
            neigh =  | 1 0 1 |
                     | 0 1 0 |
    Returns
    -------
    cmap: a dictionary containing the connectivity list: cmap[label] is a list
        of all neighbors for that label
    borders: a dictionary containing the pixels on the borders between objects.
        For each object, all the pixels on the border that are in the neighborhood
        of another object (but not background), are stored in a list with elements
        of the form  ((row,column), [list of neighboring labels])
    """

    if neigh is None:
        neigh = np.array([[0, 1, 0],
                          [1, 0, 1],
                          [0, 1, 0]])

    if neigh.shape[0] != neigh.shape[1]:
        raise ValueError('Improper neighborhood specification')

    if neigh.shape[0] % 2 != 1:
        raise ValueError('Neighborhood size must be odd')

    h = neigh.shape[0] / 2

    # To be fixed: in some cases, the neighborhood is lost for objects with
    # width (or height) of 1 pixel:
    # find pixels on the edge - simple differetial operator img(i+1) - img(i)
    # edg_h = labels.copy()
    # edg_h[1:,:] = np.abs(edg_h[:-1,:] - edg_h[1:,:])
    # edg_v = labels.copy()
    # edg_v[:,1:] = np.abs(edg_v[:,:-1] - edg_v[:,1:])
    # edg = np.zeros(labels.shape, dtype=int)
    # edg = edg_h + edg_v  # find all edge pixels, does not matter the magnitude

    labels = np.pad(labels, ((h,h),(h,h)), mode='constant', constant_values=0)
    # edg    = np.pad(edg,    ((h,h),(h,h)), mode='constant', constant_values=0)

    (n,m) = labels.shape

    cmap = {}                          # connectivity map
    borders = {}                       # borders

    # idx = np.where(edg > 0)
    idx = np.where(labels > 0)
    for (r,c) in zip(idx[0], idx[1]):        # for all points on the boundaries
        roi = labels[max(0, r-h):min(n, r+h+1), max(0, c-h):min(m, c+h+1)]
        lb  = roi[neigh == 1]
        # labels around current point, excluding its own label
        lb = np.setdiff1d(np.unique(roi), [labels[r,c]], assume_unique=True)
        if labels[r,c] in cmap:
            cmap[labels[r,c]] = np.union1d(cmap[labels[r,c]], lb)
        else:
            cmap[labels[r,c]] = lb

        lb = lb.tolist()
        if 0 in lb:                    # remove bkg
            lb.remove(0)
        if len(lb) == 0:               # no other neighbors for this pixel
            continue
        # otherwise, add all neighboring objects:
        if labels[r,c] not in borders:
            # this is a new object to add: initialize with an empty list
            # then append the info (in any case)
            borders[labels[r,c]] = list()
        borders[labels[r,c]].append( ((r,c), lb) )

    return cmap, borders
# end connectivity


# GET_SMALLEST_AREA
def get_smallest_area(reg_props):
    """
    GET_SMALLEST_AREA: returns the smallest area and the label of the corresponding
    object,

    area, label = get_smallest_area(reg_props)

    Parameters
    ----------
    reg_props: list
        a list with region properties as returned by skimage.measure.regionprops()

    Return
    ------
    area: int
        the smallest area of the objects in the list of properties
    label: int
        the label of the corresponding object; in case there are several object with
        the same area, just the first encountered one is returned

    See also
    --------
    skimage.measure.regionprops()
    """

    a = [_x['area'] for _x in reg_props ]
    l = [_x['label'] for _x in reg_props ]

    i = np.argmin(a)

    return a[i], l[i]
# end get_smallest_area


# GET_LARGEST_AREA
def get_largest_area(reg_props):
    """
    GET_LARGEST_AREA: returns the largest area and the label of the corresponding
    object,

    area, label = get_largest_area(reg_props)

    Parameters
    ----------
    reg_props: list
        a list with region properties as returned by skimage.measure.regionprops()

    Return
    ------
    area: int
        the largest area of the objects in the list of properties
    label: int
        the label of the corresponding object; in case there are several object with
        the same area, just the first encountered one is returned

    See also
    --------
    skimage.measure.regionprops()
    """

    a = [_x['area'] for _x in reg_props ]
    l = [_x['label'] for _x in reg_props ]

    i = np.argmax(a)

    return a[i], l[i]
# end get_smallest_area
