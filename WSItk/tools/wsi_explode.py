#!/usr/bin/env python2
"""
Given a huge image, this program scales and crops it into smaller images that
could be opened and processed by tools designed to handle only smaller (with
width or height < 2^32 pixels) images. It creates a hierarchy of folders (one
per selected scale), each containing cropped images from the original one. The
subimages can overlap, so that border conditions could be enforced when
applying filters.
"""
from __future__ import print_function, division, with_statement

__autor__ = 'Vlad Popovici'
__version__ = 0.1

from vipsCC import VImage
import argparse as opt
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import numpy as np

img_file   = ''
res_prefix = ''
res_format = ''
s_factors  = []
n_splits   = []
ovlap      = 0


def run():
    global img_file, res_prefix, s_factors, n_splits, ovlap

    print(img_file)
    print(res_prefix)
    print(s_factors)
    print(n_splits)
    print(ovlap)

    r = ET.Element('meta')
    t = ET.SubElement(r, 'file')
    t.text = img_file

    t = ET.SubElement(r, 'parameters')
    t1 = ET.SubElement(t, 'prefix')
    t1.text = res_prefix

    t1 = ET.SubElement(t, 'shrink')
    for s in s_factors:
        t2 = ET.SubElement(t1, 'factor')
        t2.text = str(s)

    t1 = ET.SubElement(t, 'split')
    for s in n_splits:
        t2 = ET.SubElement(t1, 'tile')
        t2.text = str(s)

    t1 = ET.SubElement(t, 'overlap')
    t1.text = str(ovlap)

    img = VImage.VImage(img_file)

    t1 = ET.SubElement(r, 'original')
    t2 = ET.SubElement(t1, 'width')
    t2.text = str(img.Xsize())
    t2 = ET.SubElement(t1, 'height')
    t2.text = str(img.Ysize())
    t2 = ET.SubElement(t1, 'channels')
    t2.text = str(img.Bands())
    t2 = ET.SubElement(t1, 'xres')
    t2.text = str(img.Xres())
    t2 = ET.SubElement(t1, 'yres')
    t2.text = str(img.Yres())
    t2 = ET.SubElement(t1, 'scale')
    t2.text = '1.0'
    t2 = ET.SubElement(t1, 'tile')
    t2.text = '(1, 1)'

    path = res_prefix + '/' + os.path.basename(img_file)
    if os.path.exists(path):
        print('Warning: Overwriting old files!')
    else:
        os.mkdir(path)

    k = 0
    for f in s_factors:
        pt = path + '/' + str(f)
        if not os.path.exists(pt):
            os.mkdir(pt)

        t1 = ET.SubElement(r, 'version')
        t2 = ET.SubElement(t1, 'scale')
        t2.text = str(f)

        n_horiz, n_vert = n_splits[k]
        t2 = ET.SubElement(t1, 'split')
        t2.text = str((n_horiz, n_vert))

        img_scaled = img.shrink(f, f)
        width = img_scaled.Xsize()
        height = img_scaled.Ysize()

        w = width / n_horiz
        h = height / n_vert
        ov = ovlap/100.0/ 2.0  # ovlap is in % and we need half of it
        sv = int(n_vert != 1)
        sh = int(n_horiz != 1)

        print('Processing scale %d and tile %d,%d' % (f, n_horiz, n_vert))

        y0 = 0
        for i in np.arange(n_vert):
            x0 = 0

            if i < n_vert - 1:
                y1 = int(y0 + h * (1.0 + sv*ov)) - 1
            else:
                y1 = height - 1

            for j in np.arange(n_horiz):
                if j < n_horiz - 1:
                    x1 = int(x0 + w * (1.0 + sh*ov)) - 1
                else:
                    x1 = width - 1

                tile_name = 'tile_' + str(i) + '_' + str(j) + '.' + res_format
                res_file = pt + '/' + tile_name
                print('Save to' + res_file)
                t2 = ET.SubElement(t1, 'tile',
                    {'name': tile_name, 'x0':str(x0), 'y0':str(y0), 'x1':str(x1), 'y1':str(y1)})

                #print('x0 = %d, y0 = %d, x1 = %d, y1 = %d, sh = %d, ov = %f; image: %d x %d' % (x0, y0, x1, y1, sh, ov, width, height))

                # do the actual work...
                img_sub = img_scaled.extract_area(x0, y0, x1-x0+1, y1-y0+1)
                img_sub.write(res_file)

                x0 = int(x1 + 1 - 2.0 * w * ov)

            y0 = int(y1 + 1 - 2.0 * w * ov)

        k += 1


    raw_txt = ET.tostring(r, 'utf-8')
    reparsed = minidom.parseString(raw_txt)
    pp_txt = reparsed.toprettyxml(indent='  ')

    #print(pp_txt)
    meta_file = open(path+'/meta.xml', 'w')
    meta_file.write(pp_txt)


def main():
    global img_file, res_prefix, s_factors, n_splits, ovlap, res_format

    p = opt.ArgumentParser(description="Generate a series of re-scaled and cropped versions of the original slide.")
    p.add_argument('img_file', action='store', help='image file')
    p.add_argument('--prefix', action='store', help='path where to store the results', default='./')
    p.add_argument('--shrink', action='store', help='shrinking factors', default='1')
    p.add_argument('--split', action='store', help='number of splits of each image on horizontal and vertical axes', default='(1,1)')
    p.add_argument('--overlap', action='store', help='overlapping percentage between images', type=float, default=0)
    p.add_argument('--format', action='store', help='output image format',
        choices=['ppm','tiff','jpeg'], default='ppm')

    args = p.parse_args()
    img_file = args.img_file
    res_prefix = args.prefix
    res_format = args.format


    # all non-integer scaling factors are rounded and ensured to be
    # positive
    s_factors = [int(abs(float(_f))) for _f in args.shrink.split(',')]

    # number of splits horizontally and vertically, at each shrinking level:
    rx = re.compile(r'(\d+,\d+)')
    n_splits = [(int(abs(float(h))),int(abs(float(v))))
        for h,v in [_p.split(',') for _p in rx.findall(args.split)]]

    # window overlapping
    ovlap = args.overlap

    if len(s_factors) != len(n_splits):
        raise(ValueError('There must be a 1-to-1 correspondence between shrinking factors and split specifiers'))


    run()


if __name__ == '__main__':
    main()
    