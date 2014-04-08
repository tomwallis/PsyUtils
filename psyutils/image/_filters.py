# fourier filtering functions.


def make_filter(filt_size, filt_type,
                f_peak=None, bw=None, alpha=None):
    """Function to make a range of basic filters.

    Applied in the fourier domain. Currently for square images only.
    Tom Wallis adapted it from makeFilter function by Peter Bex.

    Args:
        filt_size (int): the size of the filter image.
            must be an integer. This specifies the side length, so
            filt_size=256 gives you a 256 by 256 image.
        filt_type (string): which filter to use.
            This specifies the behaviour of the rest of the function
            (see below).
        f_peak (float): the filter peak (depends on filter; see below).
        bw (float): the filter bandwidth (see below).
        alpha (float): the exponent for a 1 / f^-alpha filer.

    Returns:
        image (float): the square filter, with the zero-frequency component
            at the centre of the image (i.e. not fftshifted).

    Filter Types:
        TODO docstring.

    Example:
        Create a log exponential filter with the default settings::
            filt = pu.image.make_filter(filt_size=256, filt_type="log_exp")
            pu.image.show_im(filt)

        Create an orientation filter with filter peak at 45 degrees and 10
        degrees of bandwidth::
            filt = pu.image.make_filter(filt_size=256, filt_type="orientation",
                                        f_peak = 45, bw = 10)
            pu.image.show_im(filt)

    See Also:
        image.make_filtered_noise()
        image.filter_image()
    """
    import numpy as np

    # check filt_size:
    filt_size = float(round(filt_size))

    radius = round(filt_size / 2.0)
    x = np.linspace((1 - radius), (filt_size - radius), num=filt_size)
    # meshgrid by default in cartesian coords:
    xx, yy = np.meshgrid(x, x)
    rad_dist = (xx**2 + yy**2) ** 0.5
    rad_dist[radius-1, radius-1] = 0.5  # avoid log / divide by zero problems.

    if filt_type is "log_exp":
        # set up default parameters:
        if f_peak is None:
            f_peak = filt_size / 4.0
        else:
            f_peak = float(f_peak)

        if bw is None:
            bw = 0.2  # bandwidth in log pixels.
        else:
            bw = float(bw)

        filt = np.exp(-((np.log(2)*(abs(np.log(rad_dist/f_peak)))**3) /
                     ((bw*np.log(2))**3)))

    elif filt_type is "1_f":
        if alpha is None:
            alpha = 1.0
        else:
            alpha = float(alpha)

        filt = rad_dist ** -alpha

    elif filt_type is "log_cosine":
        # set up default parameters:
        if f_peak is None:
            f_peak = filt_size / 4.0
        else:
            f_peak = float(f_peak)

        rad_dist = np.log2(rad_dist)
        filt = 0.5 * (1+np.cos(np.pi*(rad_dist-np.log2(f_peak))))
        filt[rad_dist > (np.log2(f_peak)+1)] = 0
        filt[rad_dist <= (np.log2(f_peak)-1)] = 0

    elif filt_type is "log_gauss":
        # set up default parameters:
        if f_peak is None:
            f_peak = filt_size / 4.0
        else:
            f_peak = float(f_peak)

        if bw is None:
            bw = 0.2  # bandwidth in log pixels.
        else:
            bw = float(bw)

        filt = np.exp(-((np.log2(rad_dist)-np.log2(f_peak))**2) / (2*(bw))**2)

    elif filt_type is "gauss":
        # set up default parameters:
        if f_peak is None:
            f_peak = filt_size / 4.0
        else:
            f_peak = float(f_peak)

        if bw is None:
            bw = 20.  # bandwidth in pixels.
        else:
            bw = float(bw)

        filt = np.exp(-((rad_dist-f_peak)**2) / (2*bw**2))

    elif filt_type is "high_pass":
        if f_peak is None:
            f_peak = filt_size / 4.0
        else:
            f_peak = float(f_peak)

        filt = np.zeros(rad_dist.shape)
        filt[rad_dist >= abs(f_peak)] = 1

    elif filt_type is "low_pass":
        if f_peak is None:
            f_peak = filt_size / 4.0
        else:
            f_peak = float(f_peak)

        filt = np.zeros(rad_dist.shape)
        filt[rad_dist <= abs(f_peak)] = 1

    elif filt_type is "orientation":
        # set up default parameters:
        if f_peak is None:
            f_peak = 0
        else:
            f_peak = float(f_peak)

        if bw is None:
            bw = 15  # bandwidth in degrees.
        else:
            bw = float(bw)

        # convert params to radians:
        f_peak = f_peak * np.pi / 180
        bw = bw * np.pi / 180

        ang_dist = np.arctan2(-yy, xx)
        sin_theta = np.sin(ang_dist)
        cos_theta = np.cos(ang_dist)

        ds = sin_theta * np.cos(f_peak) - cos_theta * np.sin(f_peak)
        dc = cos_theta * np.cos(f_peak) + sin_theta * np.sin(f_peak)
        dtheta = abs(np.arctan2(ds, dc))  # Absolute angular distance
        filt = np.exp((-dtheta**2) / (2*bw**2))  # ang filter component

        f_peak = f_peak + np.pi  # 180 deg offset in +ve TFs
        ds = sin_theta * np.cos(f_peak) - cos_theta * np.sin(f_peak)
        dc = cos_theta * np.cos(f_peak) + sin_theta * np.sin(f_peak)
        dtheta = abs(np.arctan2(ds, dc))  # Absolute angular distance
        filt = filt + np.exp((-dtheta**2) / (2*bw**2))  # ang filter

    else:
        raise ValueError(filt_type + " is not a recognised filter type...")

    filt[radius, radius] = 0.0
    return(filt)


def make_filtered_noise(filt_size, filt_type,
                        f_peak=None, bw=None, alpha=None):
    """Create a patch of filtered noise.
    This function makes a square patch of filtered noise of
    size filt_size. It is a wrapper for ``make_filter``; see that
    function's documentation for all the filters you can specify.

    Args:
        filt_size (int): size of the noise image (it will be square).
            Will be rounded to an integer if not.
        See documentation for make_filter for other arguments.

    Returns:
        image (float): a filtered noise image.

    Example::
        im = pu.image.make_filtered_noise(filt_size = 256,
            filt_type = "orientation", f_peak = 60, bw = 10)
        pu.image.show_im(im)

    """
    import numpy as np
    import scipy.fftpack as ft
    from numpy.random import rand

    filt_size = round(filt_size)

    filt = make_filter(filt_size=filt_size, filt_type=filt_type,
                       f_peak=f_peak, bw=bw, alpha=alpha)
    noise = rand(filt_size, filt_size)

    shifted_fft = ft.fftshift(ft.fft2(noise))

    filt_noise = np.real(ft.ifft2(ft.fftshift(shifted_fft * filt)))

    # scale :
    filt_noise = filt_noise / abs(filt_noise).max()

    return(filt_noise)


def filter_image(im, filt_type,
                 f_peak=None, bw=None, alpha=None):
    """ Filter a given image with one of the make_filter
    functions. Currently only works for square images.

    Args:
        im (float): the image to be filtered.
            Currently can only be square and 2d. I will write
            an extension to colour images sometime.
        See documentation for make_filter for other arguments.

    Returns:
        image (float): the filtered image.

    Example::
        im = pu.im_data.tiger_grey()
        filt_im = pu.image.filter_image(im,
            filt_type = "orientation", f_peak = 90, bw = 20)
        show_im(filt_im)


    """
    import numpy as np
    import scipy.fftpack as ft
    from skimage import img_as_float

    im = img_as_float(im)

    filt = make_filter(filt_size=im.shape[0], filt_type=filt_type,
                       f_peak=f_peak, bw=bw, alpha=alpha)

    shifted_fft = ft.fftshift(ft.fft2(im))

    filt_im = np.real(ft.ifft2(ft.fftshift(shifted_fft * filt)))

    # scale :
    filt_im = filt_im / abs(filt_im).max()

    return(filt_im)
