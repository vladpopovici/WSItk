#!/usr/bin/env python2

from __future__ import (absolute_import, division, print_function, unicode_literals)


__version__ = 0.01
__author__ = 'Vlad Popovici'

import os
import argparse as opt
import math as mh

import numpy as np
from scipy.cluster.hierarchy import *
from skimage.io import imread, imsave

from stain.he import rgb2he2
from descriptors.txtgrey import GaborDescriptor, LBPDescriptor, MFSDescriptor
from descriptors.extract import *
from util.explore import sliding_window
from util.visualization import enhance_patches


def main():
    p = opt.ArgumentParser(description="""
            Segments a number of rectangular contexts from a H&E slide. The contexts are clusters
            of similar regions of the image. The similarity is based on various textural
            descriptors.
            """)
    p.add_argument('img_file', action='store', help='RGB image file')
    p.add_argument('ctxt', action='store', help='Number of contexts to extract', type=int)
    p.add_argument('wsize', action='store', help='Size of the (square) regions', type=int)
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

    base_name = os.path.basename(args.img_file).split('.')
    if len(base_name) > 1:             # at least 1 suffix .ext
        base_name.pop()                # drop the extension
        base_name = '.'.join(base_name)  # reassemble the rest of the list into file name

    if args.prefix is not None:
        pfx = args.prefix
    else:
        pfx = base_name


    im = imread(args.img_file)
    print("Original image size:", im.shape)

    row_min = min(max(args.row_min, 0), im.shape[0]-2)
    col_min = min(max(args.col_min, 0), im.shape[1]-2)
    row_max = max(min(args.row_max, im.shape[0]-1), 0)
    col_max = max(min(args.col_max, im.shape[1]-1), 0)

    if row_max == 0:
        row_max = im.shape[0] - 1
    if col_max == 0:
        col_max = im.shape[1] - 1

    if row_max - row_min < args.wsize or col_max - col_min < args.wsize:
        raise ValueError('Window size too large for requested image size.')

    im = im[row_min:row_max+1, col_min:col_max+1, :]

    # crop the image to multiple of wsize:
    nh, nw = mh.floor(im.shape[0] / args.wsize), mh.floor(im.shape[1] / args.wsize)
    dh, dw = mh.floor((im.shape[0] - nh*args.wsize)/2), mh.floor((im.shape[1] - nw*args.wsize)/2)
    im = im[dh:dh+nh*args.wsize, dw:dw+nw*args.wsize, :]
    print("Image cropped to:", im.shape)
    imsave(pfx+'_cropped.ppm', im)

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

        dst = pdist_gabor(dsc)

        cl = average(dst)
        id = fcluster(cl, t=args.ctxt, criterion='maxclust')  # get the various contexts

        # save clustering/contexts - remember, the coordinates are in the
        # current image system which might have been cropped from the original ->
        # should add back the shift
        z1 = desc_to_matrix(dsc, desc_label)  # col 0: row_min, col 2: col_min
        z1[:, 0] += row_min + dh
        z1[:, 2] += col_min + dw
        z2 = np.matrix(id).transpose()
        z2 = np.hstack( (z2, z1) )
        np.savetxt(pfx+'_'+desc_label+'_h.dat', z2, delimiter="\t")

        # save visualizations
        for k in range(1,1+args.ctxt):
            i = np.where(id == k)[0]
            p = [dsc[j]['roi'] for j in i]
            im2 = enhance_patches(im, p)
            imsave(pfx+'_'+desc_label+'_h_'+str(k)+'.ppm', im2)

        if args.eosine:
            # repeat on E plane:
            print("------------> E plane")
            img_iterator = sliding_window(h.shape, (args.wsize,args.wsize),
                                          step=(args.wsize,args.wsize))
            dsc = get_local_desc(e, g, img_iterator, desc_label)

            dst = pdist_gabor(dsc)

            cl = average(dst)
            id = fcluster(cl, t=args.ctxt, criterion='maxclust')  # get the various contexts

            # save clustering/contexts - remember, the coordinates are in the
            # current image system which might have been cropped from the original ->
            # should add back the shift
            z1 = desc_to_matrix(dsc, desc_label)  # col 0:4 [row_min, row_max, col_min, col_max]
            z1[:, 0:2] += row_min + dh
            z1[:, 2:4] += col_min + dw
            z2 = np.matrix(id).transpose()
            z2 = np.hstack( (z2, z1) )
            np.savetxt(pfx+'_'+desc_label+'_e.dat', z2, delimiter="\t")

            # save visualizations
            for k in range(1,1+args.ctxt):
                i = np.where(id == k)[0]
                p = [dsc[j]['roi'] for j in i]
                im2 = enhance_patches(im, p)
                imsave(pfx+'_'+desc_label+'_e_'+str(k)+'.ppm', im2)

        print("OK")

    if args.haralick:
        print("---------> Haralick descriptors:")
        g = GLCMDescriptor()
        desc_label = 'haralick'

        print("------------> H plane")
        # on H-plane:
        img_iterator = sliding_window(h.shape, (args.wsize,args.wsize),
                                      step=(args.wsize,args.wsize))
        dsc = get_local_desc(h, g, img_iterator, desc_label)

        dst = pdist_gabor(dsc)

        cl = average(dst)
        id = fcluster(cl, t=args.ctxt, criterion='maxclust')  # get the various contexts

        # save clustering/contexts - remember, the coordinates are in the
        # current image system which might have been cropped from the original ->
        # should add back the shift
        z1 = desc_to_matrix(dsc, desc_label)  # col 0:4 [row_min, row_max, col_min, col_max]
        z1[:, 0:2] += row_min + dh
        z1[:, 2:4] += col_min + dw
        z2 = np.matrix(id).transpose()
        z2 = np.hstack( (z2, z1) )
        np.savetxt(pfx+'_'+desc_label+'_h.dat', z2, delimiter="\t")

        # save visualizations
        for k in range(1,1+args.ctxt):
            i = np.where(id == k)[0]
            p = [dsc[j]['roi'] for j in i]
            im2 = enhance_patches(im, p)
            imsave(pfx+'_'+desc_label+'_h_'+str(k)+'.ppm', im2)

        if args.eosine:
            # repeat on E plane:
            print("------------> E plane")
            img_iterator = sliding_window(h.shape, (args.wsize,args.wsize),
                                          step=(args.wsize,args.wsize))
            dsc = get_local_desc(e, g, img_iterator, desc_label)

            dst = pdist_gabor(dsc)

            cl = average(dst)
            id = fcluster(cl, t=args.ctxt, criterion='maxclust')  # get the various contexts

            # save clustering/contexts - remember, the coordinates are in the
            # current image system which might have been cropped from the original ->
            # should add back the shift
            z1 = desc_to_matrix(dsc, desc_label)  # col 0:4 [row_min, row_max, col_min, col_max]
            z1[:, 0:2] += row_min + dh
            z1[:, 2:4] += col_min + dw
            z2 = np.matrix(id).transpose()
            z2 = np.hstack( (z2, z1) )
            np.savetxt(pfx+'_'+desc_label+'_e.dat', z2, delimiter="\t")

            # save visualizations
            for k in range(1,1+args.ctxt):
                i = np.where(id == k)[0]
                p = [dsc[j]['roi'] for j in i]
                im2 = enhance_patches(im, p)
                imsave(pfx+'_'+desc_label+'_e_'+str(k)+'.ppm', im2)

        print("OK")

    if args.lbp:
        print("---------> LBP descriptors:")
        g = LBPDescriptor()
        desc_label = 'lbp'

        # on H-plane:
        print("------------> H plane")
        img_iterator = sliding_window(h.shape, (args.wsize,args.wsize),
                                      step=(args.wsize,args.wsize))
        dsc = get_local_desc(h, g, img_iterator, desc_label)

        dst = pdist_lbp(dsc)

        cl = average(dst)
        id = fcluster(cl, t=args.ctxt, criterion='maxclust')  # get the various contexts

        # save clustering/contexts - remember, the coordinates are in the
        # current image system which might have been cropped from the original ->
        # should add back the shift
        z1 = desc_to_matrix(dsc, desc_label)  # col 0:4 [row_min, row_max, col_min, col_max]
        z1[:, 0:2] += row_min + dh
        z1[:, 2:4] += col_min + dw
        z2 = np.matrix(id).transpose()
        z2 = np.hstack( (z2, z1) )
        np.savetxt(pfx+'_'+desc_label+'_h.dat', z2, delimiter="\t")

        # save visualizations
        for k in range(1,1+args.ctxt):
            i = np.where(id == k)[0]
            p = [dsc[j]['roi'] for j in i]
            im2 = enhance_patches(im, p)
            imsave(pfx+'_'+desc_label+'_h_'+str(k)+'.ppm', im2)

        if args.eosine:
            # repeat on E plane:
            print("------------> E plane")
            img_iterator = sliding_window(h.shape, (args.wsize,args.wsize),
                                          step=(args.wsize,args.wsize))
            dsc = get_local_desc(e, g, img_iterator, desc_label)

            dst = pdist_lbp(dsc)

            cl = average(dst)
            id = fcluster(cl, t=args.ctxt, criterion='maxclust')  # get the various contexts

            # save clustering/contexts - remember, the coordinates are in the
            # current image system which might have been cropped from the original ->
            # should add back the shift
            z1 = desc_to_matrix(dsc, desc_label)  # col 0:4 [row_min, row_max, col_min, col_max]
            z1[:, 0:2] += row_min + dh
            z1[:, 2:4] += col_min + dw
            z2 = np.matrix(id).transpose()
            z2 = np.hstack( (z2, z1) )
            np.savetxt(pfx+'_'+desc_label+'_e.dat', z2, delimiter="\t")

            # save visualizations
            for k in range(1,1+args.ctxt):
                i = np.where(id == k)[0]
                p = [dsc[j]['roi'] for j in i]
                im2 = enhance_patches(im, p)
                imsave(pfx+'_'+desc_label+'_e_'+str(k)+'.ppm', im2)

        print("OK")

    if args.mfs:
        print("---------> MFS descriptors:")
        g = MFSDescriptor()
        desc_label = 'mfs'

        # on H-plane:
        print("------------> H plane")
        img_iterator = sliding_window(h.shape, (args.wsize,args.wsize),
                                      step=(args.wsize,args.wsize))
        dsc = get_local_desc(h, g, img_iterator, desc_label)

        dst = pdist_mfs(dsc)

        cl = average(dst)
        id = fcluster(cl, t=args.ctxt, criterion='maxclust')  # get the various contexts

        # save clustering/contexts
        # save clustering/contexts - remember, the coordinates are in the
        # current image system which might have been cropped from the original ->
        # should add back the shift
        z1 = desc_to_matrix(dsc, desc_label)  # col 0:4 [row_min, row_max, col_min, col_max]
        z1[:, 0:2] += row_min + dh
        z1[:, 2:4] += col_min + dw
        z2 = np.matrix(id).transpose()
        z2 = np.hstack( (z2, z1) )
        np.savetxt(pfx+'_'+desc_label+'_h.dat', z2, delimiter="\t")

        # save visualizations
        for k in range(1,1+args.ctxt):
            i = np.where(id == k)[0]
            p = [dsc[j]['roi'] for j in i]
            im2 = enhance_patches(im, p)
            imsave(pfx+'_'+desc_label+'_h_'+str(k)+'.ppm', im2)

        if args.eosine:
            # repeat on E plane:
            print("------------> E plane")
            img_iterator = sliding_window(h.shape, (args.wsize,args.wsize),
                                          step=(args.wsize,args.wsize))
            dsc = get_local_desc(e, g, img_iterator, desc_label)

            dst = pdist_mfs(dsc)

            cl = average(dst)
            id = fcluster(cl, t=args.ctxt, criterion='maxclust')  # get the various contexts

            # save clustering/contexts - remember, the coordinates are in the
            # current image system which might have been cropped from the original ->
            # should add back the shift
            z1 = desc_to_matrix(dsc, desc_label)  # col 0:4 [row_min, row_max, col_min, col_max]
            z1[:, 0:2] += row_min + dh
            z1[:, 2:4] += col_min + dw
            z2 = np.matrix(id).transpose()
            z2 = np.hstack( (z2, z1) )
            np.savetxt(pfx+'_'+desc_label+'_e.dat', z2, delimiter="\t")

            # save visualizations
            for k in range(1,1+args.ctxt):
                i = np.where(id == k)[0]
                p = [dsc[j]['roi'] for j in i]
                im2 = enhance_patches(im, p)
                imsave(pfx+'_'+desc_label+'_e_'+str(k)+'.ppm', im2)

        print("OK")

    return
# end

if __name__ == '__main__':
    main()