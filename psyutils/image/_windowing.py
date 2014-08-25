# Functions for windowing, including images.
import numpy as np
import psyutils as pu


def cos_win_1d(size,
               ramp=None):
    """Create a vector of length ``size'' containing a 1D cosine window
    where the centre of the window is set to 1 and the ramps go down to
    zero symmetrically on either side.

    This can be useful for e.g. adjusting the alpha channel of a stimulus
    over frames so that the stimulus is smoothly ramped on and off.
    Since what's returned is a vector, ``size`` must be an integer
    or convertible to integer.

    This function will do some basic checking to see that your parameters
    make sense but it is not thorough. If parameters are provided as integers
    they will be converted to floats.

    Args:
        size (int):
            the size of the returned array.
        ramp (int or float, optional):
            the length of each ramp, in pixels. Defaults to ceil(size / 6).

    Returns:
        window (float): an np.array vector containing the windowing kernel,
        normalised 0--1.

    Example:
        Create a vector of size 200 where the value is 1 for the central
        110 samples with on and off ramps of size 45 at either end::
            import psyutils as pu
            window = cos_win_1d(size = 200, ramp = 45)
            pu.image.plot_win_1d(window)


    """
    # check parameters, set.
    size = float(round(size))

    if ramp is None:
        ramp = size / 6.0
    else:
        ramp = float(ramp)

    ramp = np.ceil(ramp)
    # do a check for params making sense:
    tot = (ramp * 2.0)

    if tot > size:
        raise ValueError("Your ramping parameters add up to " + str(tot) +
                         " but you " +
                         "asked for size " + str(size))

    y = np.ones((size))

    # create the ramps:
    up_ramp = np.sin(np.linspace(0, 1, round(ramp)) * np.pi/2.0)
    down_ramp = np.cos(np.linspace(0, 1, round(ramp)) * np.pi/2.0)

    # place into y:
    y[0:ramp] = up_ramp
    y[-ramp:] = down_ramp
    return(y)


def gaussian_2d(im_x, im_y=None,
                sd_x=None, sd_y=None,
                mid_x=None, mid_y=None,
                ori=0, padding=0):
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
        ori (float, optional):
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
                ori=30, mid_x=40)
            pu.image.show_im(gauss_win)

    """
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

    im_theta = ori * np.pi / 180.  # convert to radians.

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


def cos_win_2d(size,
               ramp=None,
               ramp_type='norm',
               **kwargs):
    """Create a circular Cosine window in a 2d numpy array.

    Args:
        size (int):
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively). If a single value is passed, the matrix will
            be square with this length.
        ramp (float or int, optional):
            The size of the ramp. Defaults to 0.1 (normalised).
        ramp_type (string, optional):
            Determines how `ramp` is interpreted. If 'norm' (default),
            `ramp` is the normalised size of *each* ramp, where the radius
            runs 0--1. So if `ramp`=0.1, it runs from 0.9 to 1 on each radius.
            If `ramp_type` is 'pixels', then each ramp is made to be
            ceil(ramp) pixels long.
        **kwargs (optional):
            keyword arguments passed to axes_polar. Passing this will change
            the intended behaviour of cos_win_2d.

    Returns:
        window (float): a 2d array containing the windowing kernel,
        normalised 0--1.

    Example:
        Make a cosine window in a square image of length 64::
            import psyutils as pu
            win = pu.image.cos_win_2d(size=64)
            pu.image.show_im(win)

        Make a cosine window with a larger ramp::
            import psyutils as pu
            win = pu.image.cos_win_2d(size=256, ramp=40)
            pu.image.show_im(win)

    """

    # Input checking on params:
    if ramp is None and ramp_type is 'norm':
        ramp = 0.1
    elif ramp is None and ramp_type is 'pixels':
        raise Warning("You asked for pixels but did not specify a ramp " +
                      "value. I'm defaulting to norm and ramp 0.1")
        ramp_type = 'norm'
        ramp = 0.1
    elif ramp_type is 'norm':
        ramp = float(ramp)
        if ramp > 1. or ramp < 0.:
            raise ValueError("Ramp as norm must be between 0 and 1.")
    elif ramp_type is 'pixels':
        ramp = float(np.ceil(ramp))

        if size < ramp*2.0:
            raise ValueError("Your ramping parameters add up to " +
                             str(ramp*2.0) +
                             " but you " +
                             "asked for size " + str(size))

    # convert a pixel-specified ramp into a normalised one:
    if ramp_type is 'pixels':
        # raise Warning("currently specifying in pixels assumes you're " +
        #               "getting a square image (i.e. size = a number)")
        ramp = ramp / size
        # since radial axis runs from 0--1 on all edges, to have this
        # number of pixels in the ramp need to multiply by 2:
        ramp = ramp * 2.0

    # call a radial axis:
    r, a = pu.image.axes_polar(size=size, **kwargs)
    # radial axis defaults to run from 0 -- 1.
    win = r.copy()
    ramp_start = (1. - ramp)
    ramp_end = 1.
    win[r < ramp_start] = 1.  # inside = 1.
    ramp_location = np.logical_and(r >= ramp_start,
                                   r <= ramp_end)
    win[ramp_location] = np.cos((win[ramp_location] - ramp_start)
                                / (ramp_end - ramp_start) * np.pi/2.)
    win[r > ramp_end] = 0.

    return(win)


def wedge_win(im_x,
              f_peak=None, bw=None):
    """ A radial wedge window (angular Gaussian).

    Args:
        im_x (int): the size of the window in pixels (only square now).
        f_peak (float): the peak orientation passband.
            Zero is rightwards (horizontal), moves counterclockwise in
            degrees.
        bw (float): the bandwidth of the angular Gaussian in degrees.

    """
    # set up default parameters:
    if f_peak is None:
        f_peak = 0.
    else:
        f_peak = float(f_peak)

    if bw is None:
        bw = 15.  # bandwidth in degrees.
    else:
        bw = float(bw)

    radius = round(im_x / 2.0)

    x = np.arange((1 - radius), (im_x - radius + 1))
    # meshgrid by default in cartesian coords:
    xx, yy = np.meshgrid(x, x)

    # convert params to radians:
    f_peak = f_peak * np.pi / 180
    bw = bw * np.pi / 180

    ang_dist = np.arctan2(-yy, xx)
    sin_theta = np.sin(ang_dist)
    cos_theta = np.cos(ang_dist)

    ds = sin_theta * np.cos(f_peak) - cos_theta * np.sin(f_peak)
    dc = cos_theta * np.cos(f_peak) + sin_theta * np.sin(f_peak)
    dtheta = abs(np.arctan2(ds, dc))  # Absolute angular distance
    win = np.exp((-dtheta**2) / (2*bw**2))  # ang filter component
    return(win)
