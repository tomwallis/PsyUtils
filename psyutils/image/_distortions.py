# functions to do spatial distortions.
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
