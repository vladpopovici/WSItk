#!/usr/bin/env python2

"""
WSI_IMPORT_SLIDE: imports a scanned slide into a structure used for image processing.
This structure is saved on the disk as a hierarchy of folders and files:

 .../original image name/
                        +---- meta.xml    <- meta data about the file
                        +---- first downsampling level/
                                           +---- tile_i_j.ppm...
                        +---- second downsampling level/
                                           +---- tile_i_j.ppm
                        etc

Example:
wsi_import_he_slide.py --shrink 1,2,4,8,16,32,64 --split "(20,20),(20,20),(10,10),(10,10),(10,10),(5,5),(1,1)"
    --overlap 0 orig/1017-04-f-1-vlp-40x-he.tif
"""

from __future__ import print_function, division, with_statement

__author__  = 'vlad'
__version__ = 0.15

from vipsCC import VImage
import argparse as opt
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import numpy as np
import gc

from segm.tissue import tissue_region_from_rgb
from segm.basic import bounding_box

from skimage.io import imsave, imread
from skimage.transform import resize


img_file   = ''
res_prefix = ''
res_format = ''
s_factors  = []
n_splits   = []
ovlap      = 0
n_levels   = 0


def run():
    global img_file, res_prefix, s_factors, n_splits, ovlap

    # print(img_file)
    # print(res_prefix)
    # print(s_factors)
    # print(n_splits)
    # print(ovlap)

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

    print("ROI detection: ")
    # Find the ROI:
    # img_scaled = img.shrink(100, 100)
    os.spawnv(os.P_WAIT, 'div100', ['./div100', img_file, path+'/small.ppm'])
    # save downscaled image - not the best way for going to Scikit Image,
    # but for large images we go through disk I/O anyway:
    # print("    -saving small version of the image")
    # img_scaled.write(path+'/small.ppm')

    print("    -read into scikit-learn")
    img_scaled = imread(path+'/small.ppm')

    # compute a minimal area based on the resolution of the image
    # -the image is 100x smaller than the original -> resolution is
    print("    -computing mask")
    xres, yres = img.Xres() / 100, img.Yres() / 100
    min_area   = 4 * min(xres, yres)   # ~4 mm**2
    mask, _ = tissue_region_from_rgb(img_scaled, min_area)

    # save the mask:
    print("    -saving mask")
    imsave(path + '/' + 'mask_div100.pbm', mask)
    t2 = ET.SubElement(t1, 'mask')
    t2.text = 'mask_div100.pbm'

    # coordinates of the ROI encompassing the objects, at 0.01 of original image
    print("    -detect ROI")
    rmin, cmin, rmax, cmax = bounding_box(mask)
    mask = mask[rmin:rmax+1, cmin:cmax+1]

    # get back to original coordinates, with an approximation of 100 pixels...
    rmin *= 100
    cmin *= 100
    rmax = min((rmax + 1) * 100, img.Ysize())
    cmax = min((cmax + 1) * 100, img.Xsize())

    t2 = ET.SubElement(t1, 'roi',
        {'xmin': str(cmin), 'ymin': str(rmin), 'xmax': str(cmax), 'ymax': str(rmax)})

    print("...end ROI detection.")

    # Save initial level 0:
    print("Crop ROI and save...")
    img_cropped = img.extract_area(cmin, rmin, cmax-cmin+1, rmax-rmin+1)
    img_cropped.write(path + '/pyramid-level_0.ppm')
    new_width, new_height = img_cropped.Xsize(), img_cropped.Ysize()
    img_cropped = None
    print("...OK")

    mask = None
    img = None                                            # done with it
    gc.collect()

    # Generate the pyramid
    t1 = ET.SubElement(r, 'pyramid')
    t2 = ET.SubElement(t1, 'level', {'value': '0', 'file': 'pyramid-level_0.ppm'})
    t2 = ET.SubElement(t1, 'level', {'value': '1', 'file': 'pyramid-level_1.ppm'})
    t2 = ET.SubElement(t1, 'level', {'value': '2', 'file': 'pyramid-level_2.ppm'})
    t2 = ET.SubElement(t1, 'level', {'value': '3', 'file': 'pyramid-level_3.ppm'})
    t2 = ET.SubElement(t1, 'level', {'value': '4', 'file': 'pyramid-level_4.ppm'})
    t2 = ET.SubElement(t1, 'level', {'value': '5', 'file': 'pyramid-level_5.ppm'})

    # call external tool:
    print("Computing pyramid...")
    os.spawnv(os.P_WAIT, 'pyr', ['./pyr', path+'/pyramid-level_0.ppm', path+'/pyramid', str(n_levels)])
    print("...done")

    k = 0
    for l in np.arange(n_levels+1):
        f = s_factors[l]
        pt = path + '/' + str(s_factors[l])
        if not os.path.exists(pt):
            os.mkdir(pt)

        t1 = ET.SubElement(r, 'version')
        t2 = ET.SubElement(t1, 'scale')
        t2.text = str(f)

        n_horiz, n_vert = n_splits[k]
        t2 = ET.SubElement(t1, 'split')
        t2.text = str((n_horiz, n_vert))

        # img_scaled = img_cropped.shrink(f, f)
        # load the corresponding level in the pyramid:
        img_scaled = VImage.VImage(path + '/pyramid-level_' + str(l) + '.ppm')
        width = img_scaled.Xsize()
        height = img_scaled.Ysize()

        w = width / n_horiz
        h = height / n_vert
        ov = ovlap/100.0/2.0  # ovlap is in % and we need half of it
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

    return


def main():
    global img_file, res_prefix, s_factors, n_splits, ovlap, res_format, n_levels

    p = opt.ArgumentParser(description="Generate a series of re-scaled and cropped versions of the original slide.")
    p.add_argument('img_file', action='store', help='image file')
    p.add_argument('--prefix', action='store', help='path where to store the results', default='./')
    p.add_argument('--levels', action='store', help='number of levels in pyramid', type=int, default=1)
    p.add_argument('--split', action='store', help='number of splits on horizontal and vertical axes at each level',
                   default='(1,1)')
    p.add_argument('--overlap', action='store', help='overlapping percentage between images', type=float, default=0)
    p.add_argument('--format', action='store', help='output image format',
                   choices=['ppm', 'tiff', 'jpeg'], default='ppm')

    args = p.parse_args()
    img_file = args.img_file
    res_prefix = args.prefix
    res_format = args.format
    n_levels = args.levels

    # all non-integer scaling factors are rounded and ensured to be
    # positive
    s_factors = [int(2**_f) for _f in np.arange(args.levels+1)]

    # number of splits horizontally and vertically, at each shrinking level:
    rx = re.compile(r'(\d+,\d+)')
    n_splits = [(int(abs(float(h))),int(abs(float(v))))
                for h, v in [_p.split(',') for _p in rx.findall(args.split)]]

    # window overlapping
    ovlap = args.overlap

    if len(s_factors) != len(n_splits):
        raise(ValueError('There must be (1+levels) split specifiers (first specifiers refers to original scale)'))

    run()

    return

if __name__ == '__main__':
    main()
