#!/usr/bin/env python2
"""
Automatic textural feature extraction from H&E-stained scans.
"""

from __future__ import (division, print_function, unicode_literals)


__version__ = 0.01
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import xml.etree.ElementTree as ET
from xml.dom import minidom

from descriptors.extract import extract_descriptors_he
from util.storage import ModelPersistence
import skimage.io

def main():
    p = opt.ArgumentParser(description="""
            Computes textural and tissue descriptors from an RGB image (of an H&E slide).
            """)
    p.add_argument('img_file', action='store', help='RGB image file')
    p.add_argument('model_file', action='store', help='Models file')

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


    desc = extract_descriptors_he(img, rgb_models)

    if args.meta:
        r = ET.Element('meta', attrib={'processor':'wsi_extract_descriptors'})
        t = ET.SubElement(r, 'file')
        t.text = img_file
        t = ET.SubElement(r, 'parameters')
        t1 = ET.SubElement(t, 'rgb_models')
        t1.text = args.model_file

        t = ET.SubElement(r, 'result')
        t1 = ET.SubElement(t, 'descriptors')

        t2 = ET.SubElement(t1, 'bin_prop_connective')
        t2.text = str(desc['bin_prop_connective'])
        t2 = ET.SubElement(t1, 'bin_prop_chromatin')
        t2.text = str(desc['bin_prop_chromatin'])
        t2 = ET.SubElement(t1, 'bin_prop_fat')
        t2.text = str(desc['bin_prop_fat'])
        t2 = ET.SubElement(t1, 'bin_compact_chromatin')
        t2.text = str(desc['bin_compact_chromatin'])
        t2 = ET.SubElement(t1, 'bin_compact_connective')
        t2.text =str(desc['bin_compact_connective'])
        t2 = ET.SubElement(t1, 'bin_compact_fat')
        t2.text = str(desc['bin_compact_fat'])

        t2 = ET.SubElement(t1, 'grey_gabor')
        t2.text = str(desc['grey_gabor'])
        t2 = ET.SubElement(t1, 'grey_lbp')
        t2.text = str(desc['grey_lbp'])
        t2 = ET.SubElement(t1, 'grey_glcm')
        t2.text = str(desc['grey_glcm'])

        t2 = ET.SubElement(t1, 'h_gabor')
        t2.text = str(desc['h_gabor'])
        t2 = ET.SubElement(t1, 'h_lbp')
        t2.text = str(desc['h_lbp'])
        t2 = ET.SubElement(t1, 'h_glcm')
        t2.text = str(desc['h_glcm'])

        t2 = ET.SubElement(t1, 'e_gabor')
        t2.text = str(desc['e_gabor'])
        t2 = ET.SubElement(t1, 'e_lbp')
        t2.text = str(desc['e_lbp'])
        t2 = ET.SubElement(t1, 'e_glcm')
        t2.text = str(desc['e_glcm'])

        raw_txt = ET.tostring(r, 'utf-8')
        reparsed = minidom.parseString(raw_txt)
        pp_txt = reparsed.toprettyxml(indent='  ')
        meta_file = open(base_name+'_desc.meta.xml', 'w')
        meta_file.write(pp_txt)

    return


if __name__ == '__main__':
    main()

