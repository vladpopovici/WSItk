"""
EXPLORE: implements various strategies for "exploring" an image: i.e. scanning an
image with a sliding window at different scales.
"""
__version__ = "0.0.1"
__author__ = 'vlad'


import numpy as np

def sliding_window(image_shape, w_size, start=(0,0), step=(1,1)):
    """Yield sub-images of the given image.

    Parameters
    ----------
    image_shape : tuple (nrows, ncols)
        Image shape (img.shape).
    w_size : tuple (width, height)
        Window size as a pair of width and height values.
    start : tuple (x0, y0)
        Top left corner of the first window. Defaults to (0,0).
    step : tuple (x_step, y_step)
        Step size for the sliding window, as a pair of horizontal
        and vertical steps. Defaults to (1,1).

    Returns
    -------
    sliding_window : generator
        Generator yielding sub-images from the original image.
        Data is not copied, rather a restricted view into the image
        is returned. Any changes to the returned view will be propagated
        to the original image.
    """

    img_h, img_w = image_shape

    if w_size[0] < 2 or w_size[1] < 2:
        raise ValueError('Window size too small.')

    if img_w < start[0] + w_size[0] or img_h < start[1] + w_size[1]:
        raise ValueError('Start position and/or window size out of image.')

    x, y = np.meshgrid(np.arange(0, img_w-w_size[0], step[0]),
                       np.arange(0, img_h-w_size[1], step[1]))

    top_left_corners = zip(x.reshape((-1,)).tolist(),
                           y.reshape((-1,)).tolist())


    for x0, y0 in top_left_corners:
        x1, y1 = x0 + w_size[0], y0 + w_size[1]
        yield (y0, y1, x0, x1)

## end sliding_window