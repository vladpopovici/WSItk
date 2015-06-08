# -*- coding: utf-8 -*-
"""
SEGM.BOT: bag-of-things related functions.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

__version__ = 0.01
__author__ = 'Vlad Popovici'

from util.explore import random_window_on_regions

def accumulate(image, roi, w_size, desc, bag_size, itw=None):
    if itw is None:
        itw = random_window_on_regions(image.shape, roi, w_size, bag_size)
    
    bag = []
    wnd = []
    for r in itw:
        wnd.append(r)
        bag.append(desc.compute(image[r[0]:r[1], r[2]:r[3]]))
        
    return bag, wnd
