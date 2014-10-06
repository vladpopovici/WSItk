#!/usr/bin/env python2

from __future__ import (absolute_import, division, print_function, unicode_literals)


__version__ = 0.01
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import xml.etree.ElementTree as ET
from xml.dom import minidom

import skimage.io

from segm.tissue import tissue_region_from_rgb


def main():
    p = opt.ArgumentParser(description="""
            Produces a mask covering the tissue region in the image.
            """)
    p.add_argument('img_file', action='store', help='RGB image file')
    p.add_argument('--prefix', action='store',
                   help='optional prefix for the result files: prefix_tissue_mask.pbm',
                   default=None)
    p.add_argument('--minarea', action='store',
                   help='object smaller than this will be removed',
                   default=150)
    p.add_argument('--gth', action='store',
                   help='if provided, indicates the threshold in the green channel',
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
        

    img = skimage.io.imread(args.img_file)
    mask, g_th = tissue_region_from_rgb(img, _min_area=args.minarea, 
                                       _g_th=args.gth)
    skimage.io.imsave(pfx+'_tissue_mask.pbm', 255*mask.astype('uint8'))
    
    if args.meta:
        r = ET.Element('meta', attrib={'processor':'wsi_mask'})
        t = ET.SubElement(r, 'file')
        t.text = args.img_file
        t = ET.SubElement(r, 'parameters')
        t1 = ET.SubElement(t, 'prefix')
        t1.text = args.prefix
        t1 = ET.SubElement(t, 'minarea')
        t1.text = str(args.minarea)
        t1 = ET.SubElement(t, 'gth')
        t1.text = str(args.gth)
        t = ET.SubElement(r, 'outfile')
        t.text = pfx + '_tissue_mask.pbm'
        t = ET.SubElement(r, 'gth_res')
        t.text = str(g_th)
    
        raw_txt = ET.tostring(r, 'utf-8')
        reparsed = minidom.parseString(raw_txt)
        pp_txt = reparsed.toprettyxml(indent='  ')
        meta_file = open(pfx+'_tissue_mask.meta.xml', 'w')
        meta_file.write(pp_txt)
        
    return
    
if __name__ == '__main__':
    main()
    