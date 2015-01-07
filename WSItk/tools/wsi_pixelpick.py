# -*- coding: utf-8 -*-
"""
PixelPick: allows user to select pixels in an image and returns their
coordinates and (R,G,B) (or whatever else space the image is in) values.

Created on Tue Mar 18 13:48:18 2014

@author: vlad
"""

__author__ = 'vlad'

import optparse
import matplotlib.pyplot as plt
from skimage.io import imread

# save the coordinates in:
row = []
col = []

def onpick(event):
    global row, col
    who = event.artist    # an ImageFigure object
    s = who.get_size()    # image dimensions

    # need to convert to image coordinates:
    row.append(int(s[0] - event.mouseevent.y - 1))  # rows from 0
    col.append(int(event.mouseevent.x))             # columns from 0

    return True

def pixelpick(img):
    '''
    (row, col, g)  = pixelpick(img)
    (row, col, r, g, b)  = pixelpick(img)

    Select a number of pixels in the image and return their coordinates
    and intensity (for gray-scale images) or R, G, B values (for color images).

    Parameters
    ----------
    img: ndarray
        a image, either (n,m) or (n,m,3) array

    Returns
    -------
    row: list
        row coordinate of selected pixels
    col: list
        column coordinate of selected pixels
    r: list
        red component of the selected pixels for color images
        (the first channel)
    g: list
        green component of the selected pixels for color images
        (the second channel); or gray-level (intensity) for single channel
        images
    b: list
        blue component of the selected pixels for color images
        (the third channel)
    '''
    fig = plt.figure()
    fig.figimage(img).set_picker(True)

    fig.canvas.mpl_connect('pick_event', onpick)
    plt.show()

    if len(img.shape) == 2:  # grey-scale image
        res = zip(row, col, img[row, col, 0].tolist())
    else:                    # 3-channel image
        res = zip(row, col, img[row, col, 0].tolist(), \
        img[row, col, 1].tolist(), \
        img[row, col, 2].tolist())

    return res

def main():
    p = optparse.OptionParser()
    p.add_option('-o', '--out', action='store', type='string', dest='outfile')
    p.add_option('-i', '--image', action='store', type='string', dest='infile')
    p.set_defaults(infile='', outfile='')
    opts, args = p.parse_args()

    res = pixelpick(imread(opts.infile))

    out = open(opts.outfile, 'w')
    for (x,y,r,g,b) in res:
            s = '%d %d %d %d %d\n' % (x,y,r,g,b)
            out.write(s)
    out.close()

if __name__ == '__main__':
    main()
