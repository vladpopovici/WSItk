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
from util.explore import sliding_window_on_regions
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
    p.add_argument('roi', action='store', help='a file with ROI coordinates (and context descriptors)')
    p.add_argument('label', action='store', help='the cluster label of interest')

    p.add_argument('--prefix', action='store',
                   help='optional prefix for the resulting files',
                   default=None)
    p.add_argument('--gabor', action='store_true',
                   help='compute Gabor descriptors and generate the corresponding contexts')
    p.add_argument('--lbp', action='store_true',
                   help='compute LBP (local binary patterns) descriptors and generate the corresponding contexts')
    p.add_argument('--mfs', action='store_true',
                   help='compute fractal descriptors and generate the corresponding contexts')
    p.add_argument('--eosine', action='store_true', help='should also Eosine component be processed?')

    p.add_argument('--scale', action='store', type=float, default=1.0,
                   help='scaling factor for ROI coordinates')


    args = p.parse_args()

    base_name = os.path.basename(args.img_file).split('.')
    if len(base_name) > 1:             # at least 1 suffix .ext
        base_name.pop()                # drop the extension
        base_name = '.'.join(base_name)  # reassemble the rest of the list into file name

    if args.prefix is not None:
        pfx = args.prefix
    else:
        pfx = base_name


    ROIs = []
    for l in file(args.roi).readlines():
        # extract the coordinates and the label from each ROI
        # (one per row):
        lb, row_min, row_max, col_min, col_max = map(lambda _x: int(float(_x)), l.split('\t')[1:5])
        row_min = int(mh.floor(row_min * args.scale))
        row_max = int(mh.floor(row_max * args.scale))
        col_min = int(mh.floor(col_min * args.scale))
        col_max = int(mh.floor(col_max * args.scale))
        if lb == args.label:
            ROIs.append([row_min, row_max, col_min, col_max])

    im = imread(args.img_file)
    print("Original image size:", im.shape)

    # get the H and E planes:
    h, e, _ = rgb2he2(im)

    if args.gabor:
        print("---------> Gabor descriptors:")
        g = GaborDescriptor()
        desc_label = 'gabor'

        print("------------> H plane")
        # on H-plane:
        img_iterator = sliding_window_on_regions(h.shape, ROIs, (args.wsize,args.wsize),
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
            img_iterator = sliding_window_on_regions(h.shape, ROIs, (args.wsize,args.wsize),
                                                     step=(args.wsize,args.wsize))
            dsc = get_local_desc(e, g, img_iterator, desc_label)

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
        img_iterator = sliding_window_on_regions(h.shape, ROIs, (args.wsize,args.wsize),
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
            img_iterator = sliding_window_on_regions(h.shape, ROIs, (args.wsize,args.wsize),
                                                     step=(args.wsize,args.wsize))
            dsc = get_local_desc(e, g, img_iterator, desc_label)

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
        img_iterator = sliding_window_on_regions(h.shape, ROIs, (args.wsize,args.wsize),
                                                 step=(args.wsize,args.wsize))
        dsc = get_local_desc(h, g, img_iterator, desc_label)

        dst = pdist_lbp(dsc)

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
            img_iterator = sliding_window_on_regions(h.shape, ROIs, (args.wsize,args.wsize),
                                                     step=(args.wsize,args.wsize))
            dsc = get_local_desc(e, g, img_iterator, desc_label)

            dst = pdist_lbp(dsc)

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
        img_iterator = sliding_window_on_regions(h.shape, ROIs, (args.wsize,args.wsize),
                                                 step=(args.wsize,args.wsize))
        dsc = get_local_desc(h, g, img_iterator, desc_label)

        dst = pdist_mfs(dsc)

        cl = average(dst)
        id = fcluster(cl, t=args.ctxt, criterion='maxclust')  # get the various contexts

        # save clustering/contexts
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
            img_iterator = sliding_window_on_regions(h.shape, ROIs, (args.wsize,args.wsize),
                                                     step=(args.wsize,args.wsize))
            dsc = get_local_desc(e, g, img_iterator, desc_label)

            dst = pdist_mfs(dsc)

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