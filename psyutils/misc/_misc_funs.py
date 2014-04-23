# miscellaneous functions


def fixation_cross(pix_per_degree=None):
    """Return a 256 square numpy array containing a rendering of the
    fixation cross recommended in Thaler et al for low dispersion and
    microsaccade rate. You could rescale this to the appropriate size (outer
    ring should be 0.6 dva in diameter and inner ring 0.2 dva).

    Reference:
        Thaler, L., Schütz, A. C., Goodale, M. A., & Gegenfurtner, K. R. (2013)
        What is the best fixation target? The effect of target shape on
        stability of fixational eye movements. Vision Research, 76(C), 31–42.
    """

    import numpy as np

    big_array = np.ones((256, 256))

    def _draw_oval(radius):
        im = np.ones((radius, radius))
        rad_dist =


    if pix_per_degree is not None:
        # do rescaling...


def radial_dist(im_x, im_y=None,
                mid_x=None, mid_y=None,
                grid_type=""):
    """Calculate the radial distance from a point in a 2d array.
    Defaults to use the centre of a square array, if only im_x is provided.

    Note that if the image size is an even number of pixels, the array will
    be slightly off-centre to ensure there is a pixel (at mid_y, mid_x)
    whose radius value is zero. If you would like instead to keep the grid
    symmetric (but not have a pixel at the exact centre), change the `grid_type`
    value to "symmetric".

    Args:
        im_x (int): the number of horizontal pixels in the image.
        im_y (int, optional): the number of vertical pixels. Defaults to
            be im_x (i.e. returns a square image).
        mid_x (int, optional): the horizontal point from which to calculate
            the radius. Defaults to the middle of the image.
        mid_y (int, optional): the vertical point from which to calculate
            the radius.

    Returns:
        image (ndarray): An image with the point (mid_y, mid_x) equal to
            0 and all other points giving the radial distance from this image.
    """
    import numpy as np

    if im_y is None:
        im_y = im_x
    else:
        im_y = int(im_y)

    if mid_x is None:
        mid_x = im_x / 2
    else:
        mid_x = int(mid_x)

    if mid_y is None:
        mid_y = im_y / 2
    else:
        mid_y = int(mid_y)

    x = np.arange()