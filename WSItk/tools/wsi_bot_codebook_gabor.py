# -*- coding: utf-8 -*-
"""
TOOLS.WSI_BOT: bag-of-things from an image series.
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

from sklearn.cluster import MiniBatchKMeans
from util.storage import ModelPersistence

from joblib import *

def worker(img_name, desc, wnd_size, nwindows):
    # consider the images already in H-intensity

    try:
        im_h = imread(img_name)
    except IOError as e:
        return None


    # -preprocessing
    #if im.ndim == 3:
    #    im_h, _ = rgb2he(im, normalize=True)
    #    im_h = equalize_adapthist(im_h)
    #    im_h = rescale_intensity(im_h, out_range=(0,255))
    #    im_h = im_h.astype(np.uint8)
    #else:
    #    return None

    # -image bag growing
    # bag for current image:
    bag = grow_bag_from_new_image(im_h, desc, (wnd_size, wnd_size), nwindows,
                                  sampling_strategy='random', discard_empty=True)

    # save the bag:
    with ModelPersistence('.'.join(img_name.split('.')[:-1])+'_bag_l0.pkl', 'c', format='pickle') as d:
        d['bag_l0'] = bag

    return bag[desc.name]
# end worker

def main():
    p = opt.ArgumentParser(description="""
            Constructs a dictionary for image representation based on Gabor wavelet local
            descriptors. The dictionary is built from a set of images given as a list in an
            input file.
            """)

    p.add_argument('img_path', action='store', help='path to image files - all images in the folder will be used')
    p.add_argument('img_ext', action='store', help='extension of the image files (e.g. "jpg" or "png") - NO DOT!')
    p.add_argument('out_file', action='store', help='resulting model file name')
    p.add_argument('codebook_size', action='store', help='codebook size', type=int)
    p.add_argument('-w', '--window', action='store', help='local window size', type=int, default=16)
    p.add_argument('-n', '--nwindows', action='store', help='maximal number of random windows to extract from a single image',
                   type=int, default=1000)

    args = p.parse_args()

    #---------
    # data
    data_path = args.img_path
    img_ext = args.img_ext
    wnd_size = args.window
    nwindows = args.nwindows

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

    desc_vectors = \
        Parallel(n_jobs=cpu_count()) \
        ( delayed(worker)(img_name, local_descriptor, wnd_size, nwindows) for img_name in img_files )

    print('Vector quantization:')
    print('-prepare...')
    X = np.vstack(desc_vectors)
    print('-cluster...')
    rng = np.random.RandomState(0)
    vq = MiniBatchKMeans(n_clusters=args.codebook_size, random_state=rng,
                         batch_size=500, compute_labels=True, verbose=False)   # vector quantizer

    vq.fit(X)
    print('OK')

    print('Saving model...', end='')
    # compute the average distance and std.dev. of the points in each cluster:
    avg_dist = np.zeros(args.codebook_size)
    sd_dist = np.zeros(args.codebook_size)
    for k in range(0, args.codebook_size):
        d = numpy.linalg.norm(X[vq.labels_ == k, :] - vq.cluster_centers_[k, :], axis=1)
        avg_dist[k] = d.mean()
        sd_dist[k] = d.std()

    with ModelPersistence(args.out_file, 'c', format='pickle') as d:
        d['codebook'] = vq
        d['avg_dist_to_centroid'] = avg_dist
        d['stddev_dist_to_centroid'] = sd_dist
        d['window_size'] = wnd_size

    print('OK')

    return
# end main()

if __name__ == '__main__':
    main()
