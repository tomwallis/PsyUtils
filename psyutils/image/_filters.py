# fourier filtering functions.
from __future__ import print_function, division
import numpy as np
import psyutils as pu


def make_filter_generic(filt_name,
                        pixel_units=True,
                        zero_mean=True,
                        **kwargs):
    """ A function intended for internal use, to remove redundancy in
    make_filter_X named functions. This will set up axes and apply
    the distribution function, with a few defaults if not specified.
    You should use the named wrapper functions (e.g. `make_filter_lowpass`),
    unless you're playing around.

    Args:
        filt_name (string):
            the name of the filter to be made. Can be one of:
            'lowpass', 'highpass',
        pixel_units (boolean):
            If True, units are in pixels of the array. This means that the
            parameters of the filter are in pixel units -- so the cutoff
            frequency is in cycles per image.
        zero_mean (boolean):
            If True, the zero-frequency component of the filter is zero,
            meaning that multiplication by an image in the fourier domain
            will return a zero-mean image. If False, the zero-frequency
            component is maintained, meaning the filtered image will have the
            same mean as the original.
        **kwargs:
            A number of named (keyword) arguments for passing to the axis limits
            or distribution function. Could be things like `cutoff` or `peak`.

    Returns:
        filt (matrix):
            The filter; a 2D floating point numpy array.
    """

    # for name, value in kwargs.items():
    #     print('{0} = {1}'.format(name, value))

    # determine distribution function:
    if filt_name == "lowpass":
        f = pu.dist.lowpass(peak=kwargs['cutoff'],
                            include_border=kwargs['include_border'])

    elif filt_name == "highpass":
        f = pu.dist.highpass(peak=kwargs['cutoff'],
                             include_border=kwargs['include_border'])

    elif filt_name == "gaussian":
        f = pu.dist.gauss(peak=kwargs['peak'],
                          width=kwargs['width'])

    elif filt_name == "log_gauss":
        f = pu.dist.log_gauss(peak=kwargs['peak'],
                              width=kwargs['width'])

    elif filt_name == "logexp":
        f = pu.dist.log_exponential(peak=kwargs['peak'],
                                    width=kwargs['width'])

    elif filt_name == "log_cosine":
        f = pu.dist.log_cosine(peak=kwargs['peak'])

    elif filt_name == "alpha":
        f = pu.dist.one_over_f(alpha=kwargs['alpha'])

    elif filt_name == "ori_gaussian":
        # this function is applied below.
        'blah'
    else:
        raise ValueError(filt_name + " is not a recognised filter type...")

    # make axes:
    if pixel_units is True:
        lims = pu.image.axes_limits_in_pixels(size=kwargs['size'])
        r, a = pu.image.axes_polar(size=kwargs['size'], axes_limits=lims)
    else:
        raise Exception("Tom hasn't written not-pixel-unit code yet")

    if filt_name == "ori_gaussian":
        # this is adapted from Peter Bex's matlab code. Should one day put it
        # into a function called on angular axes, but it's not simply f(a),
        # since you need to do the angular conversion for the opposite
        # phases.

        peak_radians = kwargs['peak'] * np.pi / 180.
        width_radians = kwargs['width'] * np.pi / 180.

        sin_theta = np.sin(a)
        cos_theta = np.cos(a)

        ds = sin_theta * np.cos(peak_radians) - cos_theta * np.sin(peak_radians)
        dc = cos_theta * np.cos(peak_radians) + sin_theta * np.sin(peak_radians)
        dtheta = abs(np.arctan2(ds, dc))  # Absolute angular distance
        filt = np.exp((-dtheta**2) / (2*width_radians**2))  # ang filter 1

        if kwargs['symmetric'] is True:
            peak_radians += np.pi  # add 180 deg offset for other lobe
            ds = sin_theta * np.cos(peak_radians) - cos_theta * np.sin(peak_radians)
            dc = cos_theta * np.cos(peak_radians) + sin_theta * np.sin(peak_radians)
            dtheta = abs(np.arctan2(ds, dc))  # Absolute angular distance
            filt += np.exp((-dtheta**2) / (2*width_radians**2))  # ang filter 2

    else:
        filt = f(r)

    filt = filt.astype(np.float)

    if zero_mean is True:
        filt[filt.shape[0]//2, filt.shape[1]//2] = 0.

    return(filt)


def make_filter_lowpass(size, cutoff, include_border=True,
                        pixel_units=True,
                        zero_mean=True):
    """ Make a low pass filter, which is a threshold applied to radial
    distance axes. This function is internally a wrapper for
    `make_filter_generic`.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively). If a scalar is given, result is square.
        cutoff:
            the cutoff frequency of the filter.
        include_border (boolean):
            If True, the cutoff frequency is included in the results, if False
            it is excluded.
        pixel_units (boolean):
            If True, units are in pixels of the array. This means that the
            parameters of the filter are in pixel units -- so the cutoff
            frequency is in cycles per image.
        zero_mean (boolean):
            If True, the zero-frequency component of the filter is zero,
            meaning that multiplication by an image in the fourier domain
            will return a zero-mean image. If False, the zero-frequency
            component is maintained, meaning the filtered image will have the
            same mean as the original.

    Returns:
        filt (matrix):
            The filter; a 2D floating point numpy array.

    Example:
        # a lowpass filter of 64 pixels square whose cutoff is 8
        # (radial distance in pixels, and therefore cycles per image):
        filt = pu.image.make_filter_lowpass(64, cutoff=8)

        # filter an image with it:
        im = np.random.uniform(size=(64, 64))
        im2 = pu.image.filter_image(im, filt)
        show_im(im2)

    """

    filt = pu.image.make_filter_generic(filt_name='lowpass',
                                        pixel_units=pixel_units,
                                        zero_mean=zero_mean,
                                        size=size,
                                        cutoff=cutoff,
                                        include_border=include_border)
    return(filt)


def make_filter_highpass(size, cutoff, include_border=True,
                         pixel_units=True,
                         zero_mean=True):
    """ Make a high pass filter, which is a threshold applied to radial
    distance axes. This function is internally a wrapper for
    `make_filter_generic`.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively). If a scalar is given, result is square.
        cutoff:
            the cutoff frequency of the filter.
        include_border (boolean):
            If True, the cutoff frequency is included in the results, if False
            it is excluded.
        pixel_units (boolean):
            If True, units are in pixels of the array. This means that the
            parameters of the filter are in pixel units -- so the cutoff
            frequency is in cycles per image.
        zero_mean (boolean):
            If True, the zero-frequency component of the filter is zero,
            meaning that multiplication by an image in the fourier domain
            will return a zero-mean image. If False, the zero-frequency
            component is maintained, meaning the filtered image will have the
            same mean as the original.

    Returns:
        a (matrix):
            The filter; a 2D floating point numpy array.

    Example:
        # a highpass filter of 64 pixels square whose cutoff is 8
        # (radial distance in pixels, and therefore cycles per image):
        filt = pu.image.make_filter_highpass(64, cutoff=8)

        # filter an image with it:
        im = np.random.uniform(size=(64, 64))
        im2 = pu.image.filter_image(im, filt)
        show_im(im2)

    """

    filt = pu.image.make_filter_generic(filt_name='highpass',
                                        pixel_units=pixel_units,
                                        zero_mean=zero_mean,
                                        size=size,
                                        cutoff=cutoff,
                                        include_border=include_border)
    return(filt)


def make_filter_gaussian(size, peak, width,
                         pixel_units=True,
                         zero_mean=True):
    """ Make a gaussian frequency filter, which is a gaussian
    distribution applied to radial distance axes. This function is
    internally a wrapper for `make_filter_generic`.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively). If a scalar is given, result is square.
        peak:
            the peak frequency of the filter (mean). If pixel_units is True,
            will be in pixels from the centre (i.e. corresponds to cycles
            per image of the filter peak).
        width:
            the width of the filter (sd).
        pixel_units (boolean):
            If True, units are in pixels of the array. This means that the
            parameters of the filter are in pixel units -- so the peak
            frequency is in cycles per image.
        zero_mean (boolean):
            If True, the zero-frequency component of the filter is zero,
            meaning that multiplication by an image in the fourier domain
            will return a zero-mean image. If False, the zero-frequency
            component is maintained, meaning the filtered image will have the
            same mean as the original.

    Returns:
        a (matrix):
            The filter; a 2D floating point numpy array.

    Example:
        # a gaussian filter of 64 pixels square:
        filt = pu.image.make_filter_gaussian(64, peak=8, width=2)

        # filter an image with it:
        im = np.random.uniform(size=(64, 64))
        im2 = pu.image.filter_image(im, filt)
        show_im(im2)

    """
    filt = pu.image.make_filter_generic(filt_name='gaussian',
                                        pixel_units=pixel_units,
                                        zero_mean=zero_mean,
                                        size=size,
                                        peak=peak,
                                        width=width)
    return(filt)


def make_filter_log_gauss(size, peak, width,
                          pixel_units=True,
                          zero_mean=True):
    """ Make a log-gaussian frequency filter, which is a log-gaussian
    distribution applied to radial distance axes. This function is
    internally a wrapper for `make_filter_generic`.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively). If a scalar is given, result is square.
        peak:
            the peak frequency of the filter (mean). If pixel_units is True,
            will be in pixels from the centre (i.e. corresponds to cycles
            per image of the filter peak).
        width:
            the width of the filter (log sd) -- TODO check units.
        pixel_units (boolean):
            If True, units are in pixels of the array. This means that the
            parameters of the filter are in pixel units -- so the peak
            frequency is in cycles per image.
        zero_mean (boolean):
            If True, the zero-frequency component of the filter is zero,
            meaning that multiplication by an image in the fourier domain
            will return a zero-mean image. If False, the zero-frequency
            component is maintained, meaning the filtered image will have the
            same mean as the original.

    Returns:
        a (matrix):
            The filter; a 2D floating point numpy array.

    Example:
        # a log gaussian filter of 64 pixels square:
        filt = pu.image.make_filter_log_gauss(64, peak=8, width=.2)

        # filter an image with it:
        im = np.random.uniform(size=(64, 64))
        im2 = pu.image.filter_image(im, filt)
        show_im(im2)

    """
    filt = pu.image.make_filter_generic(filt_name='log_gauss',
                                        pixel_units=pixel_units,
                                        zero_mean=zero_mean,
                                        size=size,
                                        peak=peak,
                                        width=width)
    return(filt)


def make_filter_log_exp(size, peak, width,
                        pixel_units=True,
                        zero_mean=True):
    """ Make a log exponential frequency filter, which is a log exponential
    distribution applied to radial distance axes. This function is
    internally a wrapper for `make_filter_generic`.

    The distribution is a rearranged version of equation 1 in
    Bex (2010), (In) sensitivity to spatial distortion in natural scenes.
    Journal of Vision.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively). If a scalar is given, result is square.
        peak:
            the peak frequency of the filter. If pixel_units is True, will be in
            pixels from the centre (i.e. corresponds to cycles per image of the
            filter peak).
        width:
            the half-bandwidth of the filter in octaves. So, width=0.5 gives a
            filter with a 1 octave full width.
        pixel_units (boolean):
            If True, units are in pixels of the array. This means that the
            parameters of the filter are in pixel units -- so the peak
            frequency is in cycles per image.
        zero_mean (boolean):
            If True, the zero-frequency component of the filter is zero,
            meaning that multiplication by an image in the fourier domain
            will return a zero-mean image. If False, the zero-frequency
            component is maintained, meaning the filtered image will have the
            same mean as the original.

    Returns:
        a (matrix):
            The filter; a 2D floating point numpy array.

    Example:
        # a logexp filter of 64 pixels square:
        filt = pu.image.make_filter_logexp(64, peak=8, width=0.2)

        # filter an image with it:
        im = np.random.uniform(size=(64, 64))
        im2 = pu.image.filter_image(im, filt)
        show_im(im2)

    """
    filt = pu.image.make_filter_generic(filt_name='logexp',
                                        pixel_units=pixel_units,
                                        zero_mean=zero_mean,
                                        size=size,
                                        peak=peak,
                                        width=width)
    return(filt)


def make_filter_log_cosine(size, peak,
                           pixel_units=True,
                           zero_mean=True):
    """ Make a log cosine frequency filter, which is a log cosine
    distribution applied to radial distance axes. This function is
    internally a wrapper for `make_filter_generic`.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively). If a scalar is given, result is square.
        peak:
            the peak frequency of the filter. If pixel_units is True, will be in
            pixels from the centre (i.e. corresponds to cycles per image of the
            filter peak).
        pixel_units (boolean):
            If True, units are in pixels of the array. This means that the
            parameters of the filter are in pixel units -- so the peak
            frequency is in cycles per image.
        zero_mean (boolean):
            If True, the zero-frequency component of the filter is zero,
            meaning that multiplication by an image in the fourier domain
            will return a zero-mean image. If False, the zero-frequency
            component is maintained, meaning the filtered image will have the
            same mean as the original.

    Returns:
        a (matrix):
            The filter; a 2D floating point numpy array.

    Example:
        # a log cosine filter of 64 pixels square:
        filt = pu.image.make_filter_cosine(64, peak=8)

        # filter an image with it:
        im = np.random.uniform(size=(64, 64))
        im2 = pu.image.filter_image(im, filt)
        show_im(im2)

    """
    filt = pu.image.make_filter_generic(filt_name='log_cosine',
                                        pixel_units=pixel_units,
                                        zero_mean=zero_mean,
                                        size=size,
                                        peak=peak)
    return(filt)


def make_filter_alpha_over_f(size, alpha,
                             pixel_units=True,
                             zero_mean=True):
    """ Make an alpha-over-f filter, where frequency falls off with log slope
    of alpha. If alpha=1, this is a 1/f filter. This function is
    internally a wrapper for `make_filter_generic`.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively). If a scalar is given, result is square.
        alpha:
            the negative log-log slope of power falloff. If alpha=1, this
            gives a 1/f filter.
        pixel_units (boolean):
            If True, units are in pixels of the array. This means that the
            parameters of the filter are in pixel units -- so the peak
            frequency is in cycles per image.
        zero_mean (boolean):
            If True, the zero-frequency component of the filter is zero,
            meaning that multiplication by an image in the fourier domain
            will return a zero-mean image. If False, the zero-frequency
            component is maintained, meaning the filtered image will have the
            same mean as the original.

    Returns:
        a (matrix):
            The filter; a 2D floating point numpy array.

    Example:
        # a 1/f filter of 64 pixels square:
        filt = pu.image.make_filter_alpha_over_f(64, alpha=1)

        # filter an image with it:
        im = np.random.uniform(size=(64, 64))
        im2 = pu.image.filter_image(im, filt)
        show_im(im2)

    """
    filt = pu.image.make_filter_generic(filt_name='alpha',
                                        pixel_units=pixel_units,
                                        zero_mean=zero_mean,
                                        size=size,
                                        alpha=alpha)
    return(filt)


def make_filter_orientation_gaussian(size, peak, width,
                                     symmetric=True,
                                     pixel_units=True,
                                     zero_mean=True):
    """ Make a gaussian orientation filter, which is a gaussian
    distribution applied to angular distance axes. This function is
    internally a wrapper for `make_filter_generic`.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively). If a scalar is given, result is square.
        peak:
            the peak frequency of the filter (mean) in degrees. The effects
            herein are described according to what happens to the image
            after filtering,  rather than the appearance of the filter itself.
            0 is vertical, 45 is "oblique up and left", 90 is horizontal, 135
            gives "up and right".
        width:
            the width of the filter (sd in degrees).
        symmetric:
            should the returned filter be symmetric ("bowtie") shaped? If True,
            returned filter is a bowtie, and orientations of opposite polarity
            are pooled. If "False", returned filter is a "wedge" and angles
            need to run 0-360 to get all directions.
        pixel_units (boolean):
            If True, units are in pixels of the array. This means that the
            parameters of the filter are in pixel units -- so the peak
            frequency is in cycles per image.
        zero_mean (boolean):
            If True, the zero-frequency component of the filter is zero,
            meaning that multiplication by an image in the fourier domain
            will return a zero-mean image. If False, the zero-frequency
            component is maintained, meaning the filtered image will have the
            same mean as the original.

    Returns:
        a (matrix):
            The filter; a 2D floating point numpy array.

    Example:
        # a gaussian orientation filter of 64 pixels square:
        filt = pu.image.make_filter_gaussian(64, peak=0, width=20)

        # filter an image with it:
        im = np.random.uniform(size=(64, 64))
        im2 = pu.image.filter_image(im, filt)
        show_im(im2)

    """
    filt = pu.image.make_filter_generic(filt_name='ori_gaussian',
                                        symmetric=symmetric,
                                        pixel_units=pixel_units,
                                        zero_mean=zero_mean,
                                        size=size,
                                        peak=peak,
                                        width=width)
    return(filt)


# def make_filter(im_x, filt_type,
#                 f_peak=None, bw=None, alpha=None):
#     """Function to make a range of basic filters.

#     Applied in the fourier domain. Currently for square images only.
#     Tom Wallis adapted it from makeFilter Matlab function by Peter Bex.

#     Args:
#         im_x (int): the size of the filter image.
#             must be an integer. This specifies the side length, so
#             im_x=256 gives you a 256 by 256 image.
#         filt_type (string): which filter to use.
#             This specifies the behaviour of the rest of the function
#             (see below).
#         f_peak (float): the filter peak (depends on filter; see below).
#         bw (float): the filter bandwidth (see below).
#         alpha (float): the exponent for a 1 / f^-alpha filer.

#     Returns:
#         image (float): the square filter, with the zero-frequency component
#             at the centre of the image (i.e. not fftshifted).

#     Filter Types:
#         TODO docstring.

#     Example:
#         Create a log exponential filter with the default settings::
#             filt = pu.image.make_filter(im_x=256, filt_type="log_exp")
#             pu.image.show_im(filt)

#         Create an orientation filter with filter peak at 45 degrees and 10
#         degrees of bandwidth::
#             filt = pu.image.make_filter(im_x=256, filt_type="orientation",
#                                         f_peak = 45, bw = 10)
#             pu.image.show_im(filt)

#     See Also:
#         image.make_filtered_noise()
#         image.filter_image()
#     """

#     # check im_x:
#     im_x = float(round(im_x))
#     radius = round(im_x / 2.0)
#     x = np.linspace(- radius, radius, num=im_x)
#     # meshgrid by default in cartesian coords:
#     xx, yy = np.meshgrid(x, x)
#     rad_dist = (xx**2 + yy**2) ** 0.5
#     rad_dist[radius-1, radius-1] = 0.5  # avoid log / divide by zero problems.

#     if filt_type is "log_exp":
#         # set up default parameters:
#         if f_peak is None:
#             f_peak = im_x / 4.0
#         else:
#             f_peak = float(f_peak)

#         if bw is None:
#             bw = 0.2  # bandwidth in log pixels.
#         else:
#             bw = float(bw)

#         filt = np.exp(-((np.log(2)*(abs(np.log(rad_dist/f_peak)))**3) /
#                      ((bw*np.log(2))**3)))

#     elif filt_type is "1_f":
#         if alpha is None:
#             alpha = 1.0
#         else:
#             alpha = float(alpha)

#         filt = rad_dist ** -alpha

#     elif filt_type is "log_cosine":
#         # set up default parameters:
#         if f_peak is None:
#             f_peak = im_x / 4.0
#         else:
#             f_peak = float(f_peak)

#         rad_dist = np.log2(rad_dist)
#         filt = 0.5 * (1+np.cos(np.pi*(rad_dist-np.log2(f_peak))))
#         filt[rad_dist > (np.log2(f_peak)+1)] = 0
#         filt[rad_dist <= (np.log2(f_peak)-1)] = 0

#     elif filt_type is "log_gauss":
#         # set up default parameters:
#         if f_peak is None:
#             f_peak = im_x / 4.0
#         else:
#             f_peak = float(f_peak)

#         if bw is None:
#             bw = 0.2  # bandwidth in log pixels.
#         else:
#             bw = float(bw)

#         filt = np.exp(-((np.log2(rad_dist)-np.log2(f_peak))**2) / (2*(bw))**2)

#     elif filt_type is "gauss":
#         # set up default parameters:
#         if f_peak is None:
#             f_peak = im_x / 4.0
#         else:
#             f_peak = float(f_peak)

#         if bw is None:
#             bw = 20.  # bandwidth in pixels.
#         else:
#             bw = float(bw)

#         filt = np.exp(-((rad_dist-f_peak)**2) / (2*bw**2))

#     elif filt_type is "high_pass":
#         if f_peak is None:
#             f_peak = im_x / 4.0
#         else:
#             f_peak = float(f_peak)

#         filt = np.zeros(rad_dist.shape)
#         filt[rad_dist >= abs(f_peak)] = 1

#     elif filt_type is "low_pass":
#         if f_peak is None:
#             f_peak = im_x / 4.0
#         else:
#             f_peak = float(f_peak)

#         filt = np.zeros(rad_dist.shape)
#         filt[rad_dist <= abs(f_peak)] = 1

#     elif filt_type is "orientation":
#         # set up default parameters:
#         if f_peak is None:
#             f_peak = 0
#         else:
#             f_peak = float(f_peak)

#         if bw is None:
#             bw = 15  # bandwidth in degrees.
#         else:
#             bw = float(bw)

#         # convert params to radians:
#         f_peak = f_peak * np.pi / 180
#         bw = bw * np.pi / 180

#         ang_dist = np.arctan2(-yy, xx)
#         sin_theta = np.sin(ang_dist)
#         cos_theta = np.cos(ang_dist)

#         ds = sin_theta * np.cos(f_peak) - cos_theta * np.sin(f_peak)
#         dc = cos_theta * np.cos(f_peak) + sin_theta * np.sin(f_peak)
#         dtheta = abs(np.arctan2(ds, dc))  # Absolute angular distance
#         filt = np.exp((-dtheta**2) / (2*bw**2))  # ang filter component

#         f_peak = f_peak + np.pi  # 180 deg offset in +ve TFs
#         ds = sin_theta * np.cos(f_peak) - cos_theta * np.sin(f_peak)
#         dc = cos_theta * np.cos(f_peak) + sin_theta * np.sin(f_peak)
#         dtheta = abs(np.arctan2(ds, dc))  # Absolute angular distance
#         filt = filt + np.exp((-dtheta**2) / (2*bw**2))  # ang filter

#     else:
#         raise ValueError(filt_type + " is not a recognised filter type...")

#     filt[radius, radius] = 0.0
#     return(filt)
