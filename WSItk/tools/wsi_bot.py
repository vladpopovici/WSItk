# -*- coding: utf-8 -*-
"""
TOOLS.WSI_BOT: bag-of-things from an image series.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__version__ = 0.1
__author__ = 'Vlad Popovici'

import os
import sys
import argparse as opt
from ConfigParser import SafeConfigParser
import ast
import numpy as np
import glob
import gc

import skimage
from skimage.io import imread
from skimage.util import img_as_bool
from skimage.morphology import remove_small_objects

from descriptors.basic import *
from descriptors.txtgrey import *
from stain.he import rgb2he2
from segm.tissue import tissue_region_from_rgb
from segm.basic import bounding_box
from segm.bot import *



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
    nwindows = parser.getint('sampler', 'nwindows')
                                    

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
                                    if n.lower() != 'norm'],
                _norm=tmp)
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
                tmp2 = 8* tmp
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
            
    print('No. of descriptors: ', len(local_descriptors))
    
    #---------
    # data
    if not parser.has_section('data'):
        raise ValueError('Section "data" is mandatory.')
    data_path = parser.get('data', 'input_path')
    img_ext = parser.get('data', 'image_type')
    res_path = parser.get('data', 'output_path')
    
    img_files = glob.glob(data_path + '/*.' + img_ext)
    if len(img_files) == 0:
        return
    
    ## Process:

    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)    # unbuferred output
    for img_name in img_files:
        print("Image: ", img_name, " ...reading... ", end='')
        im = imread(img_name)
        print("preprocessing... ", end='')
        # -preprocessing
        if im.ndim == 3:
            im_h, _, _ = rgb2he2(im)
        else:
            raise ValueError('Input image must be RGB.')
        
        # detect object region:
        # -try to load a precomputed mask:
        mask_file_name = data_path+'/mask/'+ \
            os.path.splitext(os.path.split(img_name)[1])[0]+ \
            '_tissue_mask.pbm'
        if os.path.exists(mask_file_name):
            print('(loading mask)...', end='')
            mask = imread(mask_file_name)
            mask = img_as_bool(mask)
            mask = remove_small_objects(mask, min_size=500, connectivity=1, in_place=True)
        else:
            print('(computing mask)...', end='')
            mask, _ = tissue_region_from_rgb(im, _min_area=500)
        
        row_min, col_min, row_max, col_max = bounding_box(mask)
        im_h[np.logical_not(mask)] = 0                       # make sure background is 0
        mask = None
        im = None
        im_h = im_h[row_min:row_max+1, col_min:col_max+1]

        print("growing the bag...", end='')
        # -image bag growing
        bag = None                               # bag for current image
        for d in local_descriptors:
            if bag is None:
                bag = grow_bag_from_new_image(im_h, d, wnd_size, nwindows, discard_empty=True)
            else:
                bag[d.name] = grow_bag_with_new_features(im_h, bag['regs'], d)[d.name]

        # save the results for each image, one file per descriptor
        desc_names = bag.keys()
        desc_names.remove('regs')                  # keep all keys but the regions
        # -save the ROI from the original image:
        res_file = res_path + '/' + 'roi-' + \
                   os.path.splitext(os.path.split(img_name)[1])[0] + '.dat'
        with open(res_file, 'w') as f:
            f.write('\t'.join([str(x_) for x_ in [row_min, row_max, col_min, col_max]]))
                    
        for dn in desc_names:
            res_file = res_path + '/' + dn + '_bag-' + \
                       os.path.splitext(os.path.split(img_name)[1])[0] + '.dat'
            with open(res_file, 'w') as f:
                n = len(bag[dn])                       # total number of descriptors of this type
                for i in range(n):
                    s = '\t'.join([str(x_) for x_ in bag['regs'][i]]) + '\t' + \
                        '\t'.join([str(x_) for x_ in bag[dn][i]]) + '\n'
                    f.write(s)
            
        print('OK')
        
        bag = None
        gc.collect()
        gc.collect()
# end main()    
    
if __name__ == '__main__':
    main()
    