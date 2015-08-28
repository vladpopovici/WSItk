# -*- coding: utf-8 -*-
"""
CONFIGS

Read various parameters from configuration files.

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__version__ = 0.5
__author__ = 'vlad'

from ConfigParser import SafeConfigParser
import ast
import numpy as np

from descriptors.basic import *
from descriptors.txtgrey import *


def read_local_descriptors_cfg(parser):
    """
    Given a parser of a configuration file (SafeConfigParser), this function
    reads the parameters for the specified local descriptors.

    :param parser:  SafeConfigParser
        A parser which already read the configuration file.

    :return: list
        a list of local descriptor objects
    """

    if not isinstance(parser, SafeConfigParser):
        raise RuntimeError('Passed parser must be an initialized SafeConfigParser')

    local_descriptors = []

    #---------
    # haar:
    if parser.has_section('haar'):
        tmp = True
        if parser.has_option('haar', 'norm'):
            tmp = parser.getboolean('haar', 'norm')
        if len(parser.items('haar')) == 0:
            # empty section, use defaults
            h = HaarLikeDescriptor(HaarLikeDescriptor.haars1())
        else:
            h = HaarLikeDescriptor([ast.literal_eval(v) for n, v in parser.items('haar')
                                    if n.lower() != 'norm'], _norm=tmp)
        local_descriptors.append(h)

    #---------
    # identity:
    if parser.has_section('identity'):
        local_descriptors.append(IdentityDescriptor())

    #---------
    # stats:
    if parser.has_section('stats'):
        tmp = []
        if parser.has_option('stats', 'mean') and parser.getboolean('stats', 'mean'):
            tmp.append('mean')
        if parser.has_option('stats', 'std') and parser.getboolean('stats', 'std'):
            tmp.append('std')
        if parser.has_option('stats', 'kurtosis') and parser.getboolean('stats', 'kurtosis'):
            tmp.append('kurtosis')
        if parser.has_option('stats', 'skewness') and parser.getboolean('stats', 'skewness'):
            tmp.append('skewness')
        if len(tmp) == 0:
            tmp = None
        local_descriptors.append(StatsDescriptor(tmp))

    #---------
    # hist:
    if parser.has_section('hist'):
        tmp = (0.0, 1.0)
        tmp2 = 10
        if parser.has_option('hist', 'min_max'):
            tmp = ast.literal_eval(parser.get('hist', 'min_max'))
            if type(tmp) != tuple:
                raise ValueError('"hist.min_max" specification error')
        if parser.has_option('hist', 'nbins'):
            tmp2 = parser.getint('hist', 'nbins')
        local_descriptors.append(HistDescriptor(_interval=tmp, _nbins=tmp2))

    #---------
    # HoG
    if parser.has_section('hog'):
        tmp = 9
        tmp2 = (128, 128)
        tmp3 = (4, 4)

        if parser.has_option('hog', 'norient'):
            tmp = parser.getint('hog', 'norient')
        if parser.has_option('hog', 'ppc'):
            tmp2 = ast.literal_eval(parser.get('hog', 'ppc'))
            if type(tmp2) != tuple:
                raise ValueError('"hog.ppc" specification error')
        if parser.has_option('hog', 'cpb'):
            tmp3 = ast.literal_eval(parser.get('hog', 'cpb'))
            if type(tmp3) != tuple:
                raise ValueError('"hog.cpb" specification error')
        local_descriptors.append(HOGDescriptor(_norient=tmp, _ppc=tmp2, _cpb=tmp3))

    #---------
    # LBP
    if parser.has_section('lbp'):
        tmp = 3
        tmp2 = 8*tmp
        tmp3 = 'uniform'

        if parser.has_option('lbp', 'radius'):
            tmp = parser.getint('lbp', 'radius')
        if parser.has_option('lbp', 'npoints'):
            tmp2 = parser.getint('lbp', 'npoints')
            if tmp2 == 0:
                tmp2 = 8 * tmp
        if parser.has_option('lbp', 'method'):
            tmp3 = parser.get('lbp', 'method')
        local_descriptors.append(LBPDescriptor(radius=tmp, npoints=tmp2, method=tmp3))

    #---------
    # Gabor
    if parser.has_section('gabor'):
        tmp  = np.array([0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0], dtype=np.double)
        tmp2 = np.array([3.0 / 4.0, 3.0 / 8.0, 3.0 / 16.0], dtype=np.double)
        tmp3 = np.array([1.0, 2 * np.sqrt(2.0)], dtype=np.double)

        if parser.has_option('gabor', 'theta'):
            tmp = ast.literal_eval(parser.get('gabor', 'theta'))
        if parser.has_option('gabor', 'freq'):
            tmp2 = ast.literal_eval(parser.get('gabor', 'freq'))
        if parser.has_option('gabor', 'sigma'):
            tmp3 = ast.literal_eval(parser.get('gabor', 'sigma'))
        local_descriptors.append(GaborDescriptor(theta=tmp, freq=tmp2, sigma=tmp3))

    return local_descriptors
