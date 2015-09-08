#!/usr/bin/env python2
#
# NDPA2CSV : convert Hamamatsu's annotation file into a CSV file
#

from __future__ import (absolute_import, division, print_function, unicode_literals)


__version__ = 0.1
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import xml.etree.ElementTree as ET
from xml.dom import minidom
import openslide as osl

def main():
    p = opt.ArgumentParser(description="""
            Convert NDPA annotation to image-based coordinates.
            """)
    p.add_argument('ndpi', action='store', help='Hamamatsu NDPI file')
    p.add_argument('ndpa', action='store', help='Hamamatsu NDPA annotation file corresponding to the NDPI file')
    p.add_argument('-d', '--div', action='store',
                   help='division factor from the level-0 magnification (e.g. 2 means half the full magnification)', default=1, type=int)
    p.add_argument('-o', '--out', action='store', help='stub of the result file name', default=None)
    args = p.parse_args()
    
    d = args.div
    if args.out is None:
        out = args.ndpi                # and we'll add annotation name when writing...
    else:
        out = args.out
        
    xml_file = ET.parse(args.ndpa)
    xml_root = xml_file.getroot()
    
    ndpi = osl.OpenSlide(args.ndpi)
    x_off = long(ndpi.properties['hamamatsu.XOffsetFromSlideCentre'])
    y_off = long(ndpi.properties['hamamatsu.YOffsetFromSlideCentre'])
    x_mpp = float(ndpi.properties['openslide.mpp-x'])
    y_mpp = float(ndpi.properties['openslide.mpp-y'])
    dimX0, dimY0 = ndpi.level_dimensions[0]
    
#    print("x_off =", x_off, "   y_off =", y_off)
#    print("x_mpp =", x_mpp, "   x_mpp =", y_mpp)
#    print("dimX0 =", dimX0, "   dimY0 =", dimY0)
    
    for ann in list(xml_root):
        name = ann.find('title').text
        p = ann.find('annotation')
        if p is None:
            continue
        p = p.find('pointlist')
        if p is None:
            continue
        fout = open(out + '_' + name +'.tab', 'w')
        
        for pts in list(p):
            # coords in NDPI system, relative to the center of the slide
            x = long(pts.find('x').text)
            y = long(pts.find('y').text)
            
            # convert the coordinates:
            x -= x_off                 # relative to the center of the image
            y -= y_off
            
            x /= (1000*x_mpp)          # in pixels, relative to the center
            y /= (1000*y_mpp)
            
            x = long(x + dimX0 / 2)    # in pixels, relative to UL corner
            y = long(y + dimY0 / 2)
            
            # save the coordinates
            fout.write('\t'.join([str(int(x/d)), str(int(y/d)), '\n']))
            #print(int(x/d), int(y/d))
        
        fout.close()
    
    return

            
if __name__ == '__main__':
    main()