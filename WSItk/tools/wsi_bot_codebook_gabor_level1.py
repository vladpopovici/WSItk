# -*- coding: utf-8 -*-
"""
TOOLS.WSI_BOT_CODEBOOK_GABOR_LEVEL1: bag-of-things from an image series; level-1 with Gabor-based level-0 codebook.
@author: vlad
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__version__ = 0.1
__author__ = 'Vlad Popovici'

import os
import sys
import argparse as opt
import numpy as np
import numpy.linalg
import glob
import itertools

import skimage
from skimage.io import imread
from skimage.util import img_as_bool
from skimage.exposure import equalize_adapthist, rescale_intensity

from descriptors.basic import *
from descriptors.txtgrey import *
from stain.he import rgb2he
from segm.tissue import tissue_region_from_rgb
from segm.basic import bounding_box
from segm.bot import *
from util.explore import sliding_window

from sklearn.cluster import MiniBatchKMeans
from util.storage import ModelPersistence
from util.math import dist

from Pycluster import kmedoids

from joblib import *

def worker(img_name, desc, wnd_size, l0_model):
    try:
        im_h = imread(img_name)
    except IOError as e:
        return None
# assume H-plane is given in the image
    # -preprocessing
#    if im.ndim == 3:
#        im_h, _ = rgb2he(im, normalize=True)
#        im_h = equalize_adapthist(im_h)
#        im_h = rescale_intensity(im_h, out_range=(0,255))
#        im_h = im_h.astype(np.uint8)
#    else:
#        return None


    print("...start on", img_name)
    bag = []
    wnd = []

    itw1 = sliding_window(im_h.shape, (wnd_size,wnd_size), start=(0,0), step=(wnd_size,wnd_size))
    for w1 in itw1:
        # for each "large window":
        # -divide it in windows of level-0 size
        # -classify these "small windows"
        # -build the histogram of codeblock frequencies
        wnd1 = im_h[w1[0]:w1[1], w1[2]:w1[3]]
        if wnd1.sum() < wnd_size**2/100:
            continue  # not enough non-zero pixels

        itw0 = sliding_window(wnd1.shape, (l0_model['window_size'],l0_model['window_size']),
                              start=(0,0), step=(l0_model['window_size'],l0_model['window_size']))
        ldesc = []
        for w0 in itw0:
            ldesc.append(desc.compute(wnd1[w0[0]:w0[1], w0[2]:w0[3]]))


        X = np.vstack(ldesc)
        y = l0_model['codebook'].predict(X)
        h = np.zeros(l0_model['codebook'].cluster_centers_.shape[0])  # histogram of code blocks
        for k in range(y.size):
            h[y[k]] += 1.0
        h /= y.size                                                      # frequencies
        bag.append(h)                                                    # add it to the bag
        wnd.append(w1)

    # end for all "large windows"

    with ModelPersistence('.'.join(img_name.split('.')[:-1])+'_bag_l1.pkl', 'c', format='pickle') as d:
        d['bag_l1'] = dict([('hist_l0', bag), ('regs', wnd)])

    print('...end on', img_name)
    return bag
# end worker


def worker_chisq_M(p, Q):
    r = np.zeros(Q.shape[0])
    for i in np.arange(Q.shape[0]):
        r[i] = dist.chisq(p, Q[i,:])
    return r.tolist()


def main():
    p = opt.ArgumentParser(description="""
            Constructs a dictionary for image representation based on histograms of codeblocks
            (Gabor wavelet local descriptors) over larger neighborhoods.
            The dictionary is built from a set of images given as a list in an input file.
            """)

    p.add_argument('img_path', action='store', help='path to image files - all images in the folder will be used')
    p.add_argument('img_ext', action='store', help='extension of the image files (e.g. "jpg" or "png") - NO DOT!')
    p.add_argument('l0_model', action='store', help='level-0 codebook model file')
    p.add_argument('out_file', action='store', help='resulting model file name')
    p.add_argument('codebook_size', action='store', help='codebook size', type=int)
    p.add_argument('-w', '--window', action='store', help='local window size (default: 512)', type=int, default=512)

    args = p.parse_args()

    #---------
    # data
    data_path = args.img_path
    img_ext = args.img_ext
    wnd_size = args.window


    with ModelPersistence(args.l0_model, 'r', format='pickle') as mp:
        l0_model = mp

    img_files = glob.glob(data_path + '/*.' + img_ext)
    if len(img_files) == 0:
        return

    #---------
    # Gabor
    tmp  = np.array([0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0], dtype=np.double)
    tmp2 = np.array([3.0 / 4.0, 3.0 / 8.0, 3.0 / 16.0], dtype=np.double)
    tmp3 = np.array([1.0, 2 * np.sqrt(2.0)], dtype=np.double)

    local_descriptor = GaborDescriptor(theta=tmp, freq=tmp2, sigma=tmp3)


    ## Process:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)    # unbuferred output
    desc_vectors = []                  # a list of local descriptor vectors

    print('Computing level 0 coding...')

    desc_vectors = \
        Parallel(n_jobs=cpu_count()) \
        ( delayed(worker)(img_name, local_descriptor, wnd_size, l0_model) for img_name in img_files )

    print('OK')

    print('Vector quantization:')
    print('-prepare...')
    X = np.vstack(desc_vectors)        # each row is a histogram
    np.save('X_bag_level1.dat', X)

    print('-compute pairwise distances...')
    n = X.shape[0]
    pdist = Parallel(n_jobs=cpu_count()) ( delayed(worker_chisq_M)(X[i,:], X[i+1:n,:]) for i in np.arange(0, n-1) )

    # make the list flat:
    pdist = np.array(list(itertools.chain.from_iterable(pdist)))

    #for i in np.arange(0, X.shape[0]-1):
    #    for j in np.arange(i+1, X.shape[0]):
    #        pdist.append(dist.chisq(X[i,:], X[j,:]))
    pdist = np.array(pdist)

    np.save('X_pdist_level1.data.npy', pdist)

    print('-cluster (k-medoids)...')
    meds = kmedoids(pdist, nclusters=args.codebook_size, npass=20)
    labels = np.unique(meds[0])  # also the indexes of vectors from X that became cluster centers (medoids)
    vq = {}
    vq['cluster_centers_'] = X[labels, :]
    vq['labels_'] = labels
    vq['distance'] = 'chisq'
    print('OK')

    print('Saving model...', end='')
    # compute the average distance and std.dev. of the points in each cluster:
    avg_dist = np.zeros(args.codebook_size)
    sd_dist = np.zeros(args.codebook_size)
    for k in range(0, args.codebook_size):
        idx = np.where(meds[0] == labels[k])[0]
        d = []
        for i in idx:
            d.append(dist.chisq(X[i,:], vq['cluster_centers_'][k,:]))
        avg_dist[k] = np.array(d).mean()
        sd_dist[k] = np.array(d).std()

    print('K-medoids summary:');
    print('-avg. dist: ', avg_dist)
    print('-std. dev. dist: ', sd_dist)

    with ModelPersistence(args.out_file, 'c', format='pickle') as d:
        d['codebook'] = vq
        d['avg_dist_to_centroid'] = avg_dist
        d['stddev_dist_to_centroid'] = sd_dist

    print('OK')

    return
# end main()

if __name__ == '__main__':
    main()
