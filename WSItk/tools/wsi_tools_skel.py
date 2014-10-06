#!/usr/bin/env python2

from __future__ import (absolute_import, division, print_function, unicode_literals)


__version__ = 0.01
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import xml.etree.ElementTree as ET
from xml.dom import minidom

import skimage.io
import skimage.exposure

#from stain.he import rgb2he


def main():
    p = opt.ArgumentParser(description="""
            Extracts the Haematoxylin and Eosin components from an RGB image (of an H&E slide).
            """)
    p.add_argument('img_file', action='store', help='RGB image file')
    p.add_argument('--prefix', action='store',
                   help='optional prefix for the result files: prefix_[h|e].type',
                   default=None)
    p.add_argument('--meta', action='store_true',
                   help='store meta information associated with the results')
    args = p.parse_args()
    base_name = os.path.basename(args.img_file).split('.')
    if len(base_name) > 1:             # at least 1 suffix .ext
        base_name.pop()                # drop the extension
        base_name = '.'.join(base_name)  # reassemble the rest of the list into file name

    if args.prefix is not None:
        pfx = args.prefix
    else:
        pfx = base_name

    if args.meta:
        r = ET.Element('meta', attrib={'processor':'wsi_he'})
        t = ET.SubElement(r, 'file')
        t.text = args.img_file
        t = ET.SubElement(r, 'parameters')
        t1 = ET.SubElement(t, 'prefix')
        t1.text = args.prefix
        t1 = ET.SubElement(t, 'histeq')
        t1.text = str(args.histeq)
        t = ET.SubElement(r, 'outfile')
        t.text = pfx + '_h.pgm'
        t = ET.SubElement(r, 'outfile')
        t.text = pfx + '_e.pgm'
    
        raw_txt = ET.tostring(r, 'utf-8')
        reparsed = minidom.parseString(raw_txt)
        pp_txt = reparsed.toprettyxml(indent='  ')
        meta_file = open(pfx+'.meta.xml', 'w')
        meta_file.write(pp_txt)
if __name__ == '__main__':
    main()