#!/usr/bin/env python2
"""
Try to segment a color (RGB) image (representing a scan of an H&E-stained
pathology slide) into basic tissue components: nuclei (chromatin),
connective tissue, fat.
"""
from __future__ import (division, print_function, unicode_literals)


__version__ = 0.01
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np

from util.storage import ModelPersistence
from segm.tissue import tissue_components

import skimage.io

def main():
    p = opt.ArgumentParser(description="""
            Segments an RGB image (of an H&E slide) into low-level tissue components.
            """)
    p.add_argument('img_file', action='store', help='RGB image file')
    p.add_argument('model_file', action='store', help='Models file')
    #p.add_argument('result_file', action='store', help='result RGB image')

    p.add_argument('--meta', action='store_true',
                   help='store meta information associated with the results')

    args = p.parse_args()
    img_file = args.img_file
    model_file = args.model_file

    base_name = os.path.basename(img_file).split('.')
    if len(base_name) > 1:             # at least 1 suffix .ext
        base_name.pop()                # drop the extension
        base_name = '.'.join(base_name)  # reassemble the rest of the list into file name

    img = skimage.io.imread(img_file)

    with ModelPersistence(model_file, 'r', format='pickle') as d:
        rgb_models = d['models']

    comp_map = tissue_components(img, rgb_models)

    res_image = np.zeros(img.shape, dtype=img.dtype)
    res_image[comp_map == 0, :] = (0, 0, 255)  # chromatin
    res_image[comp_map == 1, :] = (255, 0, 0)  # connective
    res_image[comp_map == 2, :] = (255, 255, 255) # fat

    skimage.io.imsave(base_name+"_comp.png", res_image)

    return


if __name__ == "__main__":
    main()