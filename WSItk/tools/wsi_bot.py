# -*- coding: utf-8 -*-
"""
TOOLS.WSI_BOT: bag-of-things from an image series.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

__version__ = 0.1
__author__ = 'Vlad Popovici'

import os
import argparse as opt
from ConfigParser import SafeConfigParser
import ast

from descriptors.basic import *
from descriptors.txtgrey import *

def main():
    p = opt.ArgumentParser(description="""
            Constructs a dictionary for image representation based on a set of specified local
            descriptors. The dictionary is built from a set of images given as a list in an
            input file.
            """)
    p.add_argument('config', action='store', help='a configuration file')
    args = p.parse_args()
    cfg_file = args.config
    
    parser = SafeConfigParser()
    parser.read(cfg_file)
    
    #---------
    # sampler:
    if not parser.has_section('sampler'):
        raise ValueError('"sampler" section is mandatory')
    if not parser.has_option('sampler', 'type'):
        raise ValueError('"sampler.type" is mandatory')
    tmp = parser.get('sampler', 'type').lower()
    if tmp not in ['random', 'sliding']:
        raise ValueError('Unkown sampling type')
    sampler_type = tmp
    if not parser.has_option('sampler', 'window_size'):
        raise ValueError('"sampler.window_size" is mandatory')
    wnd_size = ast.literal_eval(parser.get('sampler', 'window_size'))
    if type(wnd_size) != tuple:
        raise ValueError('"sampler.window_size" specification error')
    it_start = (0,0)
    it_step = (1,1)
    if sampler_type == 'sliding':
        if parser.has_option('sampler', 'start'):
            it_start = ast.literal_eval(parser.get('sampler','start'))
        if parser.has_option('sampler', 'step'):
            it_step  = ast.literal_eval(parser.get('sampler','step')) 
                                    

    print('sampler:')
    print('\ttype: ',sampler_type, '\twindow_size: ', wnd_size, '\tstart: ', it_start, '\tstep: ',  it_step)
    
    local_descriptors = []
    #---------
    # haar:
    if parser.has_section('haar'):
        tmp = True
        if parser.has_option('haar', 'norm'):
            tmp = parser.getboolean('haar', 'norm')
        if len(parser.items('haar')) == 0:
            # empty section, use defaults
            desc = HaarLikeDescriptor(HaarLikeDescriptor.haars1())
        else:
            h = HaarLikeDescriptor([ast.literal_eval(v) for _, v in parser.items('haar')], norm=tmp)
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
    
    
if __name__ == '__main__':
    main()
    