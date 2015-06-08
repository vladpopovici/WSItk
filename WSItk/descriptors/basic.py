from __future__ import (absolute_import, division, print_function, unicode_literals)

__author__ = 'vlad'
__version__ = 0.2


from abc import ABCMeta, abstractmethod


class LocalDescriptor:
    """
    Base class for all local descriptors: given a patch of the image, compute
    some feature vector.
    """
    __metaclass__= ABCMeta
    @abstractmethod
    def compute(self, image):
        pass

    @abstractmethod
    def dist(self, ft1, ft2, method=None):
        pass
# end class LocalDescriptor

