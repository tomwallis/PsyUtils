# functions to do eccentricity scaling of images.


def ecc_linear(im_x, im_y=None,
               mid_x=None, mid_y=None,
               scale=None):
    """Function to produce an image with values linearly increasing as a
    function of radial distance from the mid_x, mid_y point.
    """
    import numpy as np

    if im_y is None:
        im_y = im_x

    if mid_x is None:
        mid_x = im_x / 2.0
    if mid_y is None:
        mid_y = im_y / 2.0

    # construct a meshgrid of image coords:
    mid_x = im_x / 2.0
    mid_y = im_x / 2.0
    x = np.arange((1 - mid_x), (im_x - mid_x + 1))
    y = np.arange((1 - mid_y), (im_y - mid_y + 1))
    xx, yy = np.meshgrid(x, y)
    rad_dist = (xx**2 + yy**2) ** 0.5

    # scale to have max(abs) of 1:
    rad_dist = rad_dist / abs(rad_dist).max()

    # multiply by scale, if given:
    if scale is not None:
        scale = float(scale)
        rad_dist *= scale

    return(rad_dist)
