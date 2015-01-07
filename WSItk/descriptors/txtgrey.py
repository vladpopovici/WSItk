"""
DESCRIPTORS.TXTGREY: textural descriptors from grey-scale images.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

__version__ = 0.01
__author__ = 'Vlad Popovici'

from abc import ABCMeta, abstractmethod

import numpy as np
from numpy import dot
from scipy import ndimage as nd
from scipy.linalg import norm
from scipy.stats import entropy
from skimage.filter import gabor_kernel
from skimage.util import img_as_float
from skimage.feature.texture import greycoprops, greycomatrix,\
    local_binary_pattern

## A base class for textural descriptors.
class TexturalDescriptors:
    __metaclass__= ABCMeta
    @abstractmethod
    def compute(self, image):
        pass
    
    @abstractmethod
    def dist(self, ft1, ft2, method=None):
        pass
    
class GaborDescriptors(TexturalDescriptors):
    """
    Computes Gabor descriptors from an image. These descriptors are the means 
    and variances of the filter responses obtained by convolving an image with 
    a bank of Gabor filters.
    """
    def __init__(self, theta=np.array([0.0, np.pi/4.0, np.pi/2.0, 3.0*np.pi/4.0], 
                                      dtype=np.double), 
                 freq=np.array([3.0/4.0, 3.0/8.0, 3.0/16.0], dtype=np.double),
                 sigma=np.array([1.0, 2*np.sqrt(2.0)], dtype=np.double), 
                 normalized=True):
        """
        Initialize the Gabor kernels (only real part).
        
        Args:
            theta: numpy.ndarray (vector)
            Contains the orientations of the filter; defaults to [0, pi/4, pi/2, 3*pi/4].
            
            freq: numpy.ndarray (vector)
            The frequencies of the Gabor filter; defaults to [3/4, 3/8, 3/16].
            
            sigma: numpy.ndarray (vector)
            The sigma parameter for the Gaussian smoothing filter; defaults to [1, 2*sqrt(2)].
            
            normalized: bool
            If true, the kernels are normalized             
        """
        
        self.kernels_ = [np.real(gabor_kernel(frequency=f, theta=t, sigma_x=s, 
                                              sigma_y=s)) 
                         for f in freq for s in sigma for t in theta]
        if normalized:
            for k, krn in enumerate(self.kernels_):
                self.kernels_[k] = krn / np.sqrt((krn**2).sum())
        
        return
    
    def compute(self, image):
        """
        Compute the Gabor descriptors on the given image.
        
        Args:
            image: numpy.ndarray (.ndim=2)
            Grey scale image.
            
        Returns:
            numpy.ndarray (vector) containing the Gabor descriptors (means followed
            by the variances of the filter responses)
        """
        image = img_as_float(image)
        nk = len(self.kernels_)
        ft = np.zeros(2*nk, dtype=np.double)
        for k, krn in enumerate(self.kernels_):
            flt = nd.convolve(image, krn, mode='wrap')
            ft[k] = flt.mean()
            ft[k+nk] = flt.var()
        
        return ft
    
    @staticmethod
    def dist(ft1, ft2, method='Euclidean'):
        """
        Compute the distance between two sets of Gabor features. Possible distance types
        are:
            -Euclidean
            -cosine distance: this is not a proper distance! 
        
        """
        dm = {'euclidean' : lambda x_, y_: norm(x_-y_),
              'cosine': lambda x_, y_: dot(x_, y_) / (norm(x_)*norm(y_)) 
              }
        method = method.lower()
        if method not in dm.keys():
            raise ValueError('Unknown method')
        
        return dm[method](ft1, ft2)
    
        
## end class GaborDescriptors


class GLCMDescriptors(TexturalDescriptors):
    """
    Grey Level Co-occurrence Matrix: the image is decomposed into a number of
    non-overlapping regions, and the GLCM features are computed on each of these
    regions.
    """
    def __init__(self, wsize, dist=0.0, theta=0.0, levels=256, which=['dissimilarity', 'correlation'],
                 symmetric=True, normed=True):
        """
        Initialize GLCM.
        
        Args:
            wsize: uint
            window size: the image is decomposed into small non-overlapping regions of size
            <wsize x wsize> from which the GLCMs are computed. If the last region in a row or
            the last row in an image are smaller than the required size, then they are not
            used in computing

            dist: uint
            pair distance
            
            theta: float
            pair angle
            
            levels: uint
            number of grey levels
            
            which: string
            which features to be computed from the GLCM. See the help for
            skimage.feature.texture.greycoprops for details
            
            symmetric: bool
            consider symmetric pairs?
            
            normed: bool
            normalize the co-occurrence matrix, before computing the features?
        """
        self.wsize_ = wsize
        self.dist_ = dist
        self.theta_ = theta
        self.levels_ = levels
        self.which_feats_ = [w.lower() for w in which]
        self.symmetric_ = symmetric
        self.normed_ = normed
        
        return
    
    
    def compute(self, image):
        """
        Compute the GLCM features.
        """

        assert(image.ndim == 2)
        w, h = image.shape

        nw = int(w / self.wsize_)
        nh = int(h / self.wsize_)

        nf = len(self.which_feats_)

        ft = np.zeros((nf, nw*nh))  # features will be on rows
        k = 0
        for x in np.arange(0, nw):
            for y in np.arange(0, nh):
                x0, y0 = x * self.wsize_, y * self.wsize_
                x1, y1 = x0 + self.wsize_, y0 + self.wsize_

                glcm = greycomatrix(image[y0:y1, x0:x1],
                                    self.dist_, self.theta_, self.levels_,
                                    self.symmetric_, self.normed_)
                ft[:,k] = np.array([greycoprops(glcm, f)[0,0] for f in self.which_feats_])
                k += 1

        res = {}
        k = 0
        for f in self.which_feats_:
            res[f] = ft[k,:]
            k += 1

        return res
    
    
    def dist(self, ft1, ft2, method='kl'):
        """
        Computes the distance between two sets of GLCM features. The features are
        assumed to have been computed using the same parameters. The distance is
        based on comparing the distributions of these features.

        Args:
            ft1, ft2: dict
            each dictionary contains for each feature a vector of values computed
            from the images

            method: string
            the method used for computing the distance between the histograms of features:
            'kl' - Kullback-Leibler divergence (symmetrized by 0.5*(KL(p,q)+KL(q,p))
            'js' - Jensen-Shannon divergence: 0.5*(KL(p,m)+KL(q,m)) where m=(p+q)/2

        Returns:
            dict
            a dictionary with distances computed between pairs of features
        """
        # distance methods
        dm = {'kl': lambda x_, y_: 0.5*(entropy(x_, y_) + entropy(y_, x_)),
              'js': lambda x_, y_: 0.5*(entropy(x_, 0.5*(x_+y_))+entropy(y_,0.5*(x_+y_)))
              }


        method = method.lower()
        if method not in dm.keys():
            raise ValueError('Unknown method')

        res = {}
        for k in ft1.keys():
            if k in ft2.keys():
                # build the histograms:
                mn = min(ft1[k].min(), ft2[k].min())
                mx = max(ft1[k].max(), ft2[k].max())
                h1,_ = np.histogram(ft1[k], normed=True, bins=10, range=(mn,mx))
                h2,_ = np.histogram(ft2[k], normed=True, bins=10, range=(mn,mx))
                res[k] = dm[method](h1, h2)

        return res
## end class GLCMDescriptors


class LBPDescriptors(TexturalDescriptors):
    """
    Local Binary Pattern for texture description. A LBP descriptor set is a 
    histogram of LBPs computed from the image.
    """
    def __init__(self, radius=3, npoints=None, method='uniform'):
        """
        Initialize a LBP descriptor set. See skimage.feature.texture.local_binary_pattern
        for details on the meaning of parameters.
        
        Args:
            radius: int
            defaults to 3
            
            npoints: int
            defaults to None. If None, npoints is set to 8*radius
            
            method: string
            defaults to 'uniform'
        """
        
        self.radius_ = radius
        self.npoints_ = radius*8 if npoints is None else npoints
        self.method_ = method.lower()
        self.nhbins_ = self.npoints_ + 2
        
        return
    
    def compute(self, image):
        """
        Compute the LBP features. These features are returned as histograms of 
        LBPs.
        """
        lbp = local_binary_pattern(image, self.npoints_, self.radius_, self.method_)
        hist, _ = np.histogram(lbp, normed=True, bins=self.nhbins_, range=(0, self.nhbins_))
        
        return hist
    
    
    def dist(self, ft1, ft2, method='kl'):
        """
        Computes the distance between two sets of LBP features. The features are 
        assumed to have been computed using the same parameters. The features 
        are represented as histograms of LBPs.
        
        Args:
            ft1, ft2: numpy.ndarray (vector)
            histograms of LBPs as returned by compute()
            
            method: string
            the method used for computing the distance between the two sets of features:
            'kl' - Kullback-Leibler divergence (symmetrized by 0.5*(KL(p,q)+KL(q,p))
            'js' - Jensen-Shannon divergence: 0.5*(KL(p,m)+KL(q,m)) where m=(p+q)/2
        """
        # distance methods
        dm = {'kl': lambda x_, y_: 0.5*(entropy(x_, y_) + entropy(y_, x_)),
              'js': lambda x_, y_: 0.5*(entropy(x_, 0.5*(x_+y_))+entropy(y_,0.5*(x_+y_)))
              }
        
        
        method = method.lower()
        if method not in dm.keys():
            raise ValueError('Unknown method')
        
        return dm[method](ft1, ft2)

## end class LBPDescriptors     