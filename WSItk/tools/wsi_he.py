#!/usr/bin/env python2
"""
Haematoxylin and Eosin staining is the most common staining used for pathology
slides. This program extracts the information (intensity) corresponding to each
of the stainings, from a RGB image.
"""
from __future__ import (division, print_function, unicode_literals)


__version__ = 0.01
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

import skimage.io
import skimage.exposure

from stain.he import rgb2he


def main():
    p = opt.ArgumentParser(description="""
            Extracts the Haematoxylin and Eosin components from an RGB image (of an H&E slide).
            """)
    p.add_argument('img_file', action='store', help='RGB image file')
    p.add_argument('--prefix', action='store',
                   help='optional prefix for the result files: prefix_[h|e].type',
                   default=None)
    p.add_argument('--histeq', action='store_true',
                   help='requests for histogram equalization of the results')
    p.add_argument('--meta', action='store_true',
                   help='store meta information associated with the results')

    args = p.parse_args()
    img_file = args.img_file

    base_name = os.path.basename(img_file).split('.')
    if len(base_name) > 1:             # at least 1 suffix .ext
        base_name.pop()                # drop the extension
        base_name = '.'.join(base_name)  # reassemble the rest of the list into file name

    if args.prefix is not None:
        pfx = args.prefix
    else:
        pfx = base_name

    img = skimage.io.imread(img_file)

    img_h, img_e = rgb2he(img, normalize=True)

    if args.histeq:
        img_h = skimage.exposure.equalize_adapthist(img_h)
        img_e = skimage.exposure.equalize_adapthist(img_e)

    img_h = skimage.exposure.rescale_intensity(img_h, out_range=(0,255))
    img_e = skimage.exposure.rescale_intensity(img_e, out_range=(0,255))

    img_h = img_h.astype(np.uint8)
    img_e = img_e.astype(np.uint8)
    skimage.io.imsave(pfx + '_h.pgm', img_h)
    skimage.io.imsave(pfx + '_e.pgm', img_e)

    if args.meta:
        r = ET.Element('meta', attrib={'processor':'wsi_he'})
        t = ET.SubElement(r, 'file')
        t.text = img_file
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
        meta_file = open(pfx+'_he.meta.xml', 'w')
        meta_file.write(pp_txt)

    return


if __name__ == '__main__':
    main()
