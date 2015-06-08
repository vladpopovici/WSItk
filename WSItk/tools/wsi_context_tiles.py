#!/usr/bin/env python2

from __future__ import (absolute_import, division, print_function, unicode_literals)


__version__ = 0.01
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import math as mh
import re

import numpy as np
from skimage.io import imread

import xml.etree.ElementTree as ET

from stain.he import rgb2he2
from descriptors.txtgrey import GaborDescriptor
from descriptors.extract import *
from util.explore import sliding_window


def main():
    p = opt.ArgumentParser(description="""
            Segments a number of rectangular contexts from a H&E slide. The contexts are clusters
            of similar regions of the image. The similarity is based on various textural
            descriptors.
            """)
    p.add_argument('meta_file', action='store', help='XML file describing the structure of the imported file')
    p.add_argument('scale', action='store', help='which of the scales to be processed')
    p.add_argument('ctxt', action='store', help='number of contexts to extract', type=int)
    p.add_argument('wsize', action='store', help='size of the (square) regions', type=int)
    p.add_argument('--prefix', action='store',
                   help='optional prefix for the resulting files',
                   default=None)
    p.add_argument('--gabor', action='store_true',
                   help='compute Gabor descriptors and generate the corresponding contexts')
    p.add_argument('--lbp', action='store_true',
                   help='compute LBP (local binary patterns) descriptors and generate the corresponding contexts')
    p.add_argument('--mfs', action='store_true',
                   help='compute fractal descriptors and generate the corresponding contexts')
    p.add_argument('--haralick', action='store_true',
                   help='compute Haralick descriptors and generate the corresponding contexts')
    p.add_argument('--row_min', action='store', type=int, help='start row (rows start at 0)', default=0)
    p.add_argument('--col_min', action='store', type=int, help='start column (columns start at 0)', default=0)
    p.add_argument('--row_max', action='store', type=int, help='end row (maximum: image height-1)', default=0)
    p.add_argument('--col_max', action='store', type=int, help='end column (maximum: image width-1)', default=0)
    p.add_argument('--eosine', action='store_true', help='should also Eosine component be processed?')


    args = p.parse_args()

    xml_file = ET.parse(args.meta_file)
    xml_root = xml_file.getroot()

    # find the name of the image:
    base_name = os.path.basename(xml_root.find('file').text).split('.')
    if len(base_name) > 1:             # at least 1 suffix .ext
        base_name.pop()                # drop the extension
        base_name = '.'.join(base_name)  # reassemble the rest of the list into file name

    if args.prefix is not None:
        pfx = args.prefix
    else:
        pfx = base_name

    path = os.path.dirname(args.meta_file)

    # Check if the required scale exists:
    vrs = [_x for _x in xml_root.findall('version') if _x.find('scale').text == args.scale]
    if len(vrs) == 0:
        raise ValueError('The requested scale does not exits.')
    if len(vrs) > 1:
        raise ValueError('Inconsistency detected for the requested scale.')
    all_tiles = vrs[0].findall('tile')

    # get the info about full image:
    im_width = int(xml_root.find('original/width').text)
    im_height = int(xml_root.find('original/height').text)

    row_min = min(max(args.row_min, 0), im_height-2)
    col_min = min(max(args.col_min, 0), im_width-2)
    row_max = max(min(args.row_max, im_height-1), 0)
    col_max = max(min(args.col_max, im_width-1), 0)

    if row_max == 0:
        row_max = im_height - 1
    if col_max == 0:
        col_max = im_width - 1

    if row_max - row_min < args.wsize or col_max - col_min < args.wsize:
        raise ValueError('Window size too large for requested image size.')

    # keep only the tiles that overlap with the specified region
    tiles = [tl.attrib for tl in all_tiles if int(tl.attrib['x1']) >= col_min
             and col_max >= int(tl.attrib['x0'])
             and int(tl.attrib['y1']) >= row_min
             and row_max >= int(tl.attrib['y0'])]

    ## print("ROI covers", len(tiles), "tiles")

    # Sort the tiles from top to bottom and left to right.
    # -get all the (i,j) indices of the tiles:
    rx = re.compile(r'[_.]')
    ij = np.array([map(int, rx.split(t['name'])[1:3]) for t in tiles])
    # -find i_min, i_max, j_min and j_max. Since the tiles are consecutive
    # (on row and column), these are enough to generate the desired order:
    tile_i_min, tile_j_min = ij.min(axis=0)
    tile_i_max, tile_j_max = ij.max(axis=0)

    row_offset = 0
    for i in range(tile_i_min, tile_i_max+1):
        col_offset = 0
        for j in range(tile_j_min, tile_j_max+1):
            # double-check that tile_i_j is in the list of tiles:
            idx = map(lambda _x,_y: _x['name'] == _y, tiles,
                      len(tiles)*['tile_'+str(i)+'_'+str(j)+'.ppm'])
            if not any(idx):
                raise RuntimeError("Missing tile" + 'tile_'+str(i)+'_'+str(j)+'.ppm')
            tile = tiles[idx.index(True)]
            ## print("Current tile:", tile['name'])

            # Idea: the current tile (i,j) might need to be extended with a stripe
            # of maximum args.wsize to the left and bottom. So we load (if they
            # are available) the tiles (i,j+1), (i+1,j) and (i+1,j+1) and extend
            # the current tile...

            # a tile from the image is in <path>/<scale>/tile_i_j.ppm
            im = imread(path + '/' + str(args.scale) + '/' + tile['name'])
            tile_height, tile_width, _ = im.shape

            ## print("Tile size:", tile_height, "x", tile_width)

            # The scanning (sliding) windows will start at (row_offset, col_offset)
            # (in this tile's coordinate system). We want to have an integer number
            # of windows so, if needed (and possible) we will extend the current
            # tile with a block of pixels from the neighboring tiles.

            # number of windows on the horizontal
            need_expand_right = False
            right_pad = 0
            right_tile = None
            if j < tile_j_max:  # then we could eventually expand
                if (tile_width - col_offset) % args.wsize != 0:
                    need_expand_right = True
                    nh = int(mh.ceil((tile_width - col_offset) / args.wsize))
                    right_pad = nh*args.wsize - (tile_width - col_offset)
                    tile_name = 'tile_'+str(i)+'_'+str(j+1)+'.ppm'
                    idx = map(lambda _x,_y: _x['name'] == _y, tiles, len(tiles)*[tile_name])
                    assert(any(idx))
                    right_tile = tiles[idx.index(True)]

            # number of windows on the vertical
            need_expand_bot = False
            bot_pad = 0
            bot_tile = None
            if i < tile_i_max:
                if (tile_height - row_offset) % args.wsize != 0:
                    need_expand_bot = True
                    nv = int(mh.ceil((tile_height - row_offset) / args.wsize))
                    bot_pad = nv*args.wsize - (tile_height - row_offset)
                    tile_name = 'tile_'+str(i+1)+'_'+str(j)+'.ppm'
                    idx = map(lambda _x,_y: _x['name'] == _y, tiles, len(tiles)*[tile_name])
                    assert(any(idx))
                    bot_tile = tiles[idx.index(True)]

            ## print("Expand: right=", need_expand_right, "bottom=", need_expand_bot)
            ## print("...by: right=", right_pad, "bottom=", bot_pad, "pixels")

            rb_tile = None
            if need_expand_right and need_expand_bot:
                # this MUST exist if the right and bottom tiles above exist:
                tile_name = 'tile_'+str(i+1)+'_'+str(j+1)+'.ppm'
                idx = map(lambda _x,_y: _x['name'] == _y, tiles, len(tiles)*[tile_name])
                assert(any(idx))
                rb_tile = tiles[idx.index(True)]

            ## if right_tile is not None:
            ##     print("Expansion tile right:", right_tile['name'])
            ## if bot_tile is not None:
            ##     print("Expansion tile bottom:", bot_tile['name'])
            ## if rb_tile is not None:
            ##     print("Expansion tile bottom-right:", rb_tile['name'])

            # expand the image to the right and bottom only if there is a neighboring tile in
            # that direction
            r = 1 if right_tile is not None else 0
            b = 1 if bot_tile is not None else 0

            next_row_offset, next_col_offset = 0, 0

            if r+b > 0:  # we need to (and we can) pad the image with pixels from neighbors
                # Enlarge the image to the right and bottom:

                # The following line gives an error. (TypeError: 'unicode' object is not callable) Why?
                # im = np.pad(im, ((0, bot_pad), (0, right_pad), (0, 0)), mode='constant')
                im_tmp = np.zeros((tile_height+b*bot_pad, tile_width+r*right_pad, im.shape[2]))
                im_tmp[0:tile_height, 0:tile_width, :] = im
                im = im_tmp

                if right_tile is not None:
                    # a tile from the image is in <path>/<scale>/tile_i_j.ppm
                    im_tmp = imread(path + '/' + str(args.scale) + '/' + right_tile['name'])
                    im[0:tile_height, tile_width:tile_width+right_pad, :] = im_tmp[0:tile_height, 0:right_pad, :]
                    next_col_offset = right_pad

                if bot_tile is not None:
                    # a tile from the image is in <path>/<scale>/tile_i_j.ppm
                    im_tmp = imread(path + '/' + str(args.scale) + '/' + bot_tile['name'])
                    im[tile_height:tile_height+bot_pad, 0:tile_width, :] = im_tmp[0:bot_pad, 0:tile_width, :]
                    next_row_offset = bot_pad

                if rb_tile is not None:
                    # a tile from the image is in <path>/<scale>/tile_i_j.ppm
                    im_tmp = imread(path + '/' + str(args.scale) + '/' + rb_tile['name'])
                    im[tile_height:tile_height+bot_pad, tile_width:tile_width+right_pad, :] = im_tmp[0:bot_pad, 0:right_pad, :]

                im_tmp = None  # discard

            # From the current tile (padded), we need to process the region
            # (row_offset, col_offset) -> (im.height, im.width) (with new
            # height and width). But there might still be some restrictions
            # due to the region of interest (row_min, col_min) -> (row_max, col_max).
            # These last coordinates are in global coordinate system! So, first we
            # convert them to (rmn, cmn) -> (rmx, cmx), and lower bound them to
            # the offset:
            rmn = max(row_min - int(tile['y0']), row_offset)
            rmx = min(row_max - int(tile['y0']) + 1, im.shape[0])
            cmn = max(col_min - int(tile['x0']), col_offset)
            cmx = min(col_max - int(tile['x0']) + 1, im.shape[1])

            ## print("Final region of the image:", rmn, rmx, cmn, cmx)

            im = im[rmn:rmx, cmn:cmx, :]  # image to process

            # tile contains the real coordinates of the region in the image
            crt_row_min = int(tile['y0'])
            crt_col_min = int(tile['x0'])

            col_offset = next_col_offset

            ## print("Next offsets:", row_offset, col_offset)
            ## print("=======================================================")
            ## print("=======================================================")

            # Finally, we have the image for analysis. Don't forget to transform the coordinates
            # from current tile system to global image system when saving the results.
            if im.shape[0] < args.wsize or im.shape[1] < args.wsize:
                # (what is left of the) tile is smaller than the window size
                continue
            # get the H and E planes:
            h, e, _ = rgb2he2(im)

            if args.gabor:
                print("---------> Gabor descriptors:")
                g = GaborDescriptor()
                desc_label = 'gabor'

                print("------------> H plane")
                # on H-plane:
                img_iterator = sliding_window(h.shape, (args.wsize,args.wsize),
                                              step=(args.wsize,args.wsize))
                dsc = get_local_desc(h, g, img_iterator, desc_label)
                id = np.zeros((1, len(dsc)))   # we do not cluster here...

                # save clustering/contexts - remember, the coordinates are in the
                # current tile/image system -> should add back the shift
                z1 = desc_to_matrix(dsc, desc_label)  # col 0: row_min, col 2: col_min
                z1[:, 0:2] += crt_row_min + rmn
                z1[:, 2:4] += crt_col_min + cmn
                z2 = np.matrix(id).transpose()
                z2 = np.hstack( (z2, z1) )
                np.savetxt(pfx+'_'+tile['name']+'_'+desc_label+'_h.dat', z2, delimiter="\t")

                if args.eosine:
                    # repeat on E plane:
                    print("------------> E plane")
                    img_iterator = sliding_window(h.shape, (args.wsize,args.wsize),
                                                  step=(args.wsize,args.wsize))
                    dsc = get_local_desc(e, g, img_iterator, desc_label)
                    id = np.zeros((1, len(dsc)))   # we do not cluster here...

                    # save clustering/contexts - remember, the coordinates are in the
                    # current tile/image system -> should add back the shift
                    z1 = desc_to_matrix(dsc, desc_label)  # col 0: row_min, col 2: col_min
                    z1[:, 0:2] += crt_row_min + rmn
                    z1[:, 2:4] += crt_col_min + cmn
                    z2 = np.matrix(id).transpose()
                    z2 = np.hstack( (z2, z1) )
                    np.savetxt(pfx+'_'+tile['name']+'_'+desc_label+'_e.dat', z2, delimiter="\t")
                print("OK")

        # end for j...
        row_offset = next_row_offset
    # end for i....

    return
# end

if __name__ == '__main__':
    main()