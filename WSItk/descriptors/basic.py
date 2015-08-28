from __future__ import (absolute_import, division, print_function, unicode_literals)

__author__ = 'vlad'
__version__ = 0.3
__all__ = ['LocalDescriptor', 'IdentityDescriptor']

from abc import ABCMeta, abstractmethod
from future.utils import bytes_to_native_str as nstr
from scipy.linalg import norm
from numpy import dot


class LocalDescriptor:
    """
    Base class for all local descriptors: given a patch of the image, compute
    some feature vector.
    """
    __metaclass__ = ABCMeta
    name = nstr(b'LocalDescriptor')

    @abstractmethod
    def compute(self, image):
        pass

    @staticmethod
    def dist(self, ft1, ft2, method=''):
        return 0.0
# end class LocalDescriptor


class IdentityDescriptor(LocalDescriptor):
    """
    A dummy descriptor, allowing to treat all cases uniformly.
    This descriptor returns the local neighborhood, reformatted as
    a vector.
    """
    name = nstr(b'identity')

    def __init__(self):
        pass

    def compute(self, image):
        """
        Returns all the pixels in the region as a vector.

        :param image: numpy.array
            Image data.
        :return: numpy.ndarray 1D
        """
        return image.reshape(image.size)

    @staticmethod
    def dist(self, ft1, ft2, method='euclidean'):
        dm = {'euclidean': lambda x_, y_: norm(x_ - y_),
              'cosine': lambda x_, y_: dot(x_, y_) / (norm(x_) * norm(y_))
              }

        method = method.lower()
        if method not in dm:
            raise ValueError('Unknown method')

        return dm['method'](ft1, ft2)
# end IdentityDescriptor
