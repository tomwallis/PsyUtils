# Functions for windowing, including images.


def cos_win_1d(length,
               ramp=None,
               padding=0):
    """Create a vector of length ``length`` containing a 1D cosine window
    where the centre of the window is set to 1 and the ramps go down to
    zero symmetrically on either side (with optional zero padding).

    This can be useful for e.g. adjusting the alpha channel of a stimulus
    over frames so that the stimulus is smoothly ramped on and off.
    Since what's returned is a vector, ``length`` must be an integer
    or convertible to integer.

    This function will do some basic checking to see that your parameters
    make sense but it is not thorough. If parameters are provided as integers
    they will be converted to floats.

    Args:
        length (int):
            the length of the window function to return.
        ramp (int or float, optional):
            the size of each ramp. Defaults to length / 6.
        padding (int, optional):
            the size of zero padding at the edges. Defaults to zero.

    Returns:
        window (float): a vector containing the windowing kernel,
        normalised 0--1.

    Example:
        Create a vector of length 200 where the value is 1 for the central
        100 samples, on and off ramps of length 45 and 5 samples of zero
        padding at either end::
            import psyutils as pu
            window = cos_win_1d(length = 200, ramp = 45, padding = 5)
            pu.image.plot_win_1d(window)


    """

    import numpy as np

    # check parameters, set.
    length = float(length)
    length_int = int(length)

    if ramp is None:
        ramp = length / 6.0
    else:
        ramp = float(ramp)

    padding = int(padding)

    # do a check for params making sense:
    tot = (ramp * 2.0) + (padding * 2.0)

    if tot > length:
        raise ValueError("Your ramping parameters add up to " + str(tot) +
                         " but you " +
                         "asked for length " + str(length))

    y = np.ones((length_int))

    # create the ramps:
    up_ramp = np.sin(np.linspace(0, 1, round(ramp)) * np.pi/2.0)
    down_ramp = np.cos(np.linspace(0, 1, round(ramp)) * np.pi/2.0)

    # place into y:
    if padding is not 0:
        y[padding:(padding+ramp)] = up_ramp
        y[-(ramp + padding):-padding] = down_ramp
        y[0:padding] = 0
        y[-padding:] = 0
    else:
        y[0:ramp] = up_ramp
        y[-ramp:] = down_ramp

    return(y)


def gaussian_2d(im_x, im_y=None,
                sd_x=None, sd_y=None,
                mid_x=None, mid_y=None,
                im_ori=0, padding=0):
    """Create a Gaussian located in a 2d numpy array.
    Specifying just the required parameter im_x creates a
    symmetrical Gaussian centred in the image with the default sd
    (im_x / 6.0).

    Specifying combinations of the other parameters allows you to
    create varying locations, symmetries and orientations of Gaussian
    blobs.

    Args:
        im_x (int):
            The horizontal size of the image. If this is the only argument
            supplied then the image will be square with sd = im_x/6.
        im_y (int, optional):
            The vertical size of the image, if different to the horizontal.
        sd_x (float, optional):
            The horizontal standard deviation of the Gaussian. If only sd_x
            is supplied, the Gaussian will be symmetric (i.e. sd_y == sd_x).
        sd_y (float, optional):
            The vertical sd of the Gaussian.
        mid_x (float, optional):
            The horizontal mid-point of the Gaussian in (sub-) pixel units.
        mid_y (float, optional):
            The vertical mid-point of the Gaussian in (sub-) pixel units.
        im_ori (float, optional):
            Degrees of rotation to apply to the Gaussian (counterclockwise).
        padding (int, optional):
            the size of zero padding at the edges. Defaults to zero.
            This assumes that the Gaussian is centred (will just make the
            values a circle of zeros outside mid_x - padding).

    Returns:
        window (float): a 2d array containing the windowing kernel,
        normalised 0--1.

    Example:
        Make a symmetrical gaussian in a square image of length 64::
            import psyutils as pu
            gauss_win = pu.image.gaussian_2d(im_x=64)
            pu.image.show_im(gauss_win)

        Make a rotated, asymmetrical and off-centre Gaussian::
            import psyutils as pu
            gauss_win = pu.image.gaussian_2d(im_x=128, sd_x=10, sd_y=30,
                im_ori=30, mid_x=40)
            pu.image.show_im(gauss_win)

    """

    import numpy as np

    # setup symmetric variables, if others not supplied:
    if sd_x is None:
        sd_x = im_x / 6.0

    if sd_y is None:
        sd_y = sd_x  # symmetrical

    if im_y is None:
        im_y = im_x

    if mid_x is None and mid_y is None:
        mid_x = im_x / 2.0
        mid_y = im_y / 2.0
    elif mid_x is not None and mid_y is None:
        mid_y = im_y / 2.0
    elif mid_x is None and mid_y is not None:
        mid_x = im_x / 2.0

    im_theta = im_ori * np.pi / 180.  # convert to radians.

    x = np.arange((1 - mid_x), (im_x - mid_x + 1))
    y = np.arange((1 - mid_y), (im_y - mid_y + 1))

    # meshgrid by default in cartesian coords:
    xx, yy = np.meshgrid(x, y)
    x_rot = np.sin(im_theta) * xx + np.cos(im_theta) * yy
    y_rot = np.sin(im_theta) * yy - np.cos(im_theta) * xx
    win = np.exp(-(x_rot**2.0 / (2.0*sd_y**2.0))) * \
        np.exp(-(y_rot**2.0 / (2.0*sd_x**2.0)))

    if padding is not 0:
        xx, yy = np.meshgrid(x, y)
        rad_dist = (xx**2 + yy**2) ** 0.5
        win[rad_dist >= (mid_x-padding)] = 0

    return(win)


def cos_win_2d(im_size,
               ramp=None):
    """Create a circular Cosine window in a 2d numpy array.

    Args:
        im_size (int):
            The length of one side of the image.
        ramp (float or int, optional):
            The size of the ramp in pixels. Defaults to side length / 6.0.

    Returns:
        window (float): a 2d array containing the windowing kernel,
        normalised 0--1.

    Example:
        Make a cosine window in a square image of length 64::
            import psyutils as pu
            win = pu.image.cos_win_2d(im_size=64)
            pu.image.show_im(win)

    """

    import numpy as np

    if ramp is None:
        ramp = round(im_size / 6.0)
    else:
        ramp = float(round(ramp))

    radius = im_size / 2.0

    # do a check for params making sense:
    tot = (ramp * 2.0)

    if tot > im_size:
        raise ValueError("Your ramping parameters add up to " + str(tot) +
                         " but you " +
                         "asked for size " + str(im_size))

    x = np.arange((1 - radius), (im_size - radius + 1))

    xx, yy = np.meshgrid(x, x)
    rad_dist = (xx**2 + yy**2) ** 0.5

    win = rad_dist.copy()
    win[rad_dist <= radius-ramp/2.] = 1  # inside 1
    win[rad_dist > radius+ramp/2.] = 0  # outside 0
    ramp_location = [np.logical_and(rad_dist > (radius-ramp/2.),
                                    rad_dist < (radius+ramp/2.))]
    win[ramp_location] = 0.5 + 0.5*np.cos(np.pi / 2. + np.pi *
                                         (win[ramp_location] - radius) /
                                         (ramp))

    return(win)
def plot_win_1d(y):
    """Helper function for visualising 1d windows

    """

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, y.size, num=y.size)
    line, = plt.plot(x, y, linewidth=2)

    plt.show()
