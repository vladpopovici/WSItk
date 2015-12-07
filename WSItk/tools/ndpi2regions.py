# -*- coding: utf-8 -*-
"""
NDPI2REGIONS: mask out all the regions that are not in the list of annotations.

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)


__version__ = 0.1
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import xml.etree.ElementTree as ET
from xml.dom import minidom
import openslide as osl
import cv2
import numpy as np
from skimage.draw import polygon


def main():
    p = opt.ArgumentParser(description="""
            Convert NDPA annotation to image-based coordinates.
            """)
    p.add_argument('ndpi', action='store', help='Hamamatsu NDPI file')
    p.add_argument('ndpa', action='store', help='Hamamatsu NDPA annotation file corresponding to the NDPI file')
    p.add_argument('image', action='store', help='An image extracted from the slide file, on which the regions will be masked')
    p.add_argument('out', action='store', help='result image file name')
    p.add_argument('div', action='store',
                   help='division factor from the level-0 magnification (e.g. 2 means half the full magnification)', type=int)
    p.add_argument('-c', '--crop', action='store_true', help='Crop the image to the minimum object bounding box?')

    args = p.parse_args()

    d = args.div

    img = cv2.imread(args.image)
    mask = np.zeros((img.shape[0], img.shape[1]))

    xml_file = ET.parse(args.ndpa)
    xml_root = xml_file.getroot()

    ndpi = osl.OpenSlide(args.ndpi)
    x_off = long(ndpi.properties['hamamatsu.XOffsetFromSlideCentre'])
    y_off = long(ndpi.properties['hamamatsu.YOffsetFromSlideCentre'])
    x_mpp = float(ndpi.properties['openslide.mpp-x'])
    y_mpp = float(ndpi.properties['openslide.mpp-y'])
    dimX0, dimY0 = ndpi.level_dimensions[0]

    xmax, ymax = 0, 0                  # bounding box of all objects
    xmin, ymin = img.shape[:2]

    for ann in list(xml_root):
        name = ann.find('title').text
        p = ann.find('annotation')
        if p is None:
            continue
        if p.find('closed').text != "1":
            continue                   # not a closed contour
        p = p.find('pointlist')
        if p is None:
            continue

        xy_coords = []
        for pts in list(p):
            # coords in NDPI system, relative to the center of the slide
            x = long(pts.find('x').text)
            y = long(pts.find('y').text)

            # convert the coordinates:
            x -= x_off                 # relative to the center of the image
            y -= y_off

            x /= (1000*x_mpp)          # in pixels, relative to the center
            y /= (1000*y_mpp)

            x = long((x + dimX0 / 2)/d) # in pixels, relative to UL corner
            y = long((y + dimY0 / 2)/d)

            xy_coords.append([x, y])

        if len(xy_coords) < 5:
            # too short
            continue

        # check the last point to match the first one
        if (xy_coords[0][0] != xy_coords[-1][0]) or (xy_coords[0][1] != xy_coords[-1][1]):
            xy_coords.append(xy_coords[0])

        xy_coords = np.array(xy_coords, dtype=np.int32)
        xmn, ymn = xy_coords.min(axis=0)
        xmx, ymx = xy_coords.max(axis=0)
        xmin = min(xmn, xmin)
        ymin = min(ymn, ymin)
        xmax = max(xmx, xmax)
        ymax = max(ymx, ymax)

        #print('\t'.join([str(xmin), str(ymin), str(xmax), str(ymax)]))
        if np.any(xy_coords < 0):
            print('negative coords')
        # cv2.fillConvexPoly(mask, xy_coords, 1)
        # cv2.fillPoly(mask, xy_coords, 1)
        rr, cc = polygon(xy_coords[:,1], xy_coords[:,0])
        mask[rr, cc] = 1
        # end for ann...

    # the mask contains all the regions of interest:
    # -convert to bool
    # -negate the mask (regions not of interest become TRUE
    # -mask the image (set pixels TRUE in the mask to 0)
    mask = mask.astype(np.bool)
    if not mask.any():
        print("No ROI found. Giving up...")
        return
    if mask.all():
        print("ROI equals whole image. Nothing to do...")
        return

    mask = np.logical_not(mask)
    img[mask] = 0

    if args.crop:
        img = img[ymin:ymax+1,xmin:xmax+1]

    # save the result
    cv2.imwrite(args.out, img)

    return


if __name__ == '__main__':
    main()