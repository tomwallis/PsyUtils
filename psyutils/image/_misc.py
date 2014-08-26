# Miscellaneous functions.

import numpy as np
from scipy.interpolate import griddata


def grid_distort(im, x_offset, y_offset,
                 method="linear",
                 fill_method=None):
    """Apply x and y offsets to distort an image using 2d interpolation.

    Based on a method by Peter Bex (see ref, below).

    The x_offset and y_offset variables should be images (ndarrays) of the same
    size as im. These will be used as the x- and y- offsets for the pixels in
    scipy's griddata algorithm.

    Args:
        im (float): the image to distort.
        x_offset (float): the array of x_offset points.
        y_offset (float): the array of y_offset points.
        method (string): the interpolation method.
            This is passed to scipy.interpolate.griddata's method argument,
            and can be either "nearest", "linear", or "cubic".
        fill_method (float, optional): how to deal with NaNs in interp.
            If left as None (the default), any `nan` values from interpolation
            (caused by input points outside the convex hull) will be replaced
            by the mean of the image. If instead you specify a float, then this
            value will be inserted into the image by griddata.

    Returns:
        image (float): the distorted image.

    Example:
        Distort an image with a log cosine filter::
            im = img_as_float(pu.im_data.tiger_grey())
            filt = pu.image.make_filter(filt_size=im.shape[0],
                                        filt_type="log_cosine",
                                        f_peak = 8)
            gauss = pu.image.gaussian_2d(im_x=filt.shape[0])
            scale = 20
            filt_x = pu.image.make_filtered_noise(filt) * gauss * scale
            filt_y = pu.image.make_filtered_noise(filt) * gauss * scale
            dist_im = pu.image.grid_distort(im, x_offset=filt_x,
                                            y_offset=filt_y,
                                            method="linear")
            pu.image.show_im(dist_im)

    Reference:
        Bex, P. J. (2010). (In) sensitivity to spatial distortion in natural
        scenes. Journal of Vision, 10(2), 23:1-15.

    """
    # could also try RectBivariateSpline, but I couldn't get this to like
    # my dimensions, even with .ravel()...

    # construct a meshgrid of image coords:
    mid_x = im.shape[1] / 2.0
    mid_y = im.shape[0] / 2.0
    x = np.arange((1 - mid_x), (im.shape[1] - mid_x + 1))
    y = np.arange((1 - mid_y), (im.shape[0] - mid_y + 1))
    xx, yy = np.meshgrid(x, y)

    xx_new = xx + x_offset
    yy_new = yy + y_offset
    # might do the following as a loop over image dimensions for colour ims.
    # interpolate stuff returns a class (function) that can be later called
    # to do the interp itself:

    # what value to use for nans in interp:
    if fill_method is None:
        fill_value = im.mean()
    else:
        fill_value = float(fill_method)

    z_new = griddata(points=(xx.ravel(), yy.ravel()),
                     values=im.ravel(),
                     xi=(xx_new, yy_new),
                     method=method,
                     fill_value=fill_value)

    return(z_new)


def make_filtered_noise(filt):
    """Create a patch of filtered noise, the same size as filt.
    This function makes a patch of filtered noise of
    the same shape as filt. It is a wrapper for ``filter_image``
    that specifically uses noise. See the doc for ``make_function``
    for all the filters you can specify.

    Args:
        filt (float): a filter, centred in the fourier domain.
        See documentation for make_filter for other arguments.

    Returns:
        image (float): a filtered noise image.

    Example::
        filt = pu.image.make_filter(im_x=im.shape[0],
                                    filt_type="orientation",
                                    f_peak=90, bw=20)
        im = pu.image.make_filtered_noise(filt)
        pu.image.show_im(im)

    """
    from numpy.random import rand
    from psyutils.image import filter_image

    noise = rand(filt.shape[0], filt.shape[1])
    filt_noise = filter_image(noise, filt)
    return(filt_noise)


def filter_image(im, filt):
    """ Filter a given image with a given filter by multiplying in the
    frequency domain. Image and filter must be the same size.

    Args:
        im (float): the image to be filtered.
            Currently can only be square and 2d. I will write
            an extension to colour images sometime.
        filt (float): the filter.
            Should be centred in Fourier space (i.e. zero frequency
            component should be in the middle of the image).

    Returns:
        image (float): the filtered image. Will be scaled between -1
            and 1 (zero mean).

    Example::
        im = pu.im_data.tiger_grey()
        filt = pu.image.make_filter(im_x=im.shape[0],
                                    filt_type="orientation",
                                    f_peak=90, bw=20)
        filt_im = pu.image.filter_image(im, filt)
        show_im(filt_im)


    """
    import scipy.fftpack as ft
    from skimage import img_as_float

    im = img_as_float(im)
    shifted_fft = ft.fftshift(ft.fft2(im))
    filt_im = np.real(ft.ifft2(ft.fftshift(shifted_fft * filt)))
    # scale with max abs value of 1:
    # filt_im = filt_im / abs(filt_im).max()
    return(filt_im)
