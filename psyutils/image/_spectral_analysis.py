# functions to do a spectral analysis of a greyscale image.
#
# Some parts of this code adapted with thanks from a notebook on
# image statistics by Bart Machilsen and Maarten Demeyer (see
# http://nbviewer.jupyter.org/github/gestaltrevision/python_for_visres/blob/master/Part7/Part7_Image_Statistics.ipynb)

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import psyutils as pu


def spectral_analysis_fft(im):
    """
    Wrapper for fft; including a shift to centre the
    lowest frequency.

    Args:
        im:
            (2D numpy array). The image to return fft.
    Returns:
        A tuple of:
            A: the amplitude spectrum
            P: the phase spectrum
            F: the full (complex) spectrum

    Example:
        A, P, F = spectral_analysis_fft(im)
        show_im(np.log(A))
    """
    # Do the fft
    F = np.fft.fft2(im)

    # Center the spectrum on the lowest frequency
    F_centered = np.fft.fftshift(F)

    # Extract amplitude and phase
    A = np.abs(F_centered).real
    P = np.angle(F_centered).real

    # Return amplitude, phase, and the full spectrum
    return A, P, F


def spectral_analysis(im,
                      sf_band=2,
                      sf_step=1,
                      ori_band=10,
                      ori_step=4,
                      angular_mode="direction"):
    """Return the mean amplitude for different radii,
    averaging over angle; and for different angles
    averaging over radii. That is, returns spectral
    power as a function of spatial frequency (c/img) and
    orientation. Averaging is done using Gaussian sliding windows.

    Orientation results for the "direction" mode map directly onto
    the fft amplitude space: starting at zero at 3 o'clock ("right")
    and moving counterclockwise through 2 pi. So horizontal energy is
    horizontal modulation (i.e. "vertical" edges), vertical energy is
    *horizontal* edges, and the distribution for most images will be
    symmetrical, resulting from the approximately equal power for each
    edge polarity.

    Orientation results for the "orientation" mode average over the
    edges of opposite polarities (using a "bowtie"-shaped filter).
    The returned angles run from 0 to pi (180 degrees), where horizontal
    (0 / pi) is horizontal modulation, pi/2 is vertical modulation, etc.
    For polar plots it may be nicer to multiply this by two.

    Args:
        im:
            (2D numpy array). The image to analyse.
        sf_band:
            the bandwidth of the sliding window for frequency
            (SD of the Gaussian),
            in cycles per image (pixels).
        sf_step:
            the step size of the sliding window for frequency,
            in pixels.
        ori_band:
            the bandwidth of the angular sliding window
            (SD of the Gaussian),
            in degrees.
        ori_step:
            the step size of the angular sliding window,
            in degrees.
        angular_mode:
            Mode of the angular averaging (see above).
    Returns:
        A dict containing:
            'freq': vector of frequencies (c / img)
            'freq_amp': corresponding vector of amplitudes
            'ang': vector of angles.
            'ang_amp': corresponding vector of amplitudes.

    Example:
        spec = spectral_analysis(im)
        plt.plot(np.log(spec['freq']), np.log(spec['freq_amp']))

    """

    # check inputs
    if len(im.shape) > 2:
        raise ValueError("not currently defined for > 2D images")

    # do fft:
    A, P, F = spectral_analysis_fft(im)

    freq, freq_amp = _do_radial_scan(A, sf_band, sf_step)

    if angular_mode == "direction":
        ang, ang_amp = _do_angular_scan(A,
                                        ori_band, ori_step,
                                        symmetric=False)
    elif angular_mode == "orientation":
        ang, ang_amp = _do_angular_scan(A,
                                        ori_band, ori_step,
                                        symmetric=True)
    else:
        raise ValueError("unknown angular mode")

    return {'freq': freq,
            'freq_amp': freq_amp,
            'ang': ang,
            'ang_amp': ang_amp}


def _do_radial_scan(A, sf_band, sf_step):
    """
    Internal function for computing the radial part of the spectral analysis.
    Uses a radial Gaussian sliding window.
    """

    finished = 0
    inc = 0
    freq = []
    freq_amp = []
    while finished == 0:
        peak = 1 + inc  # start from 1 cpi
        freq.append(peak)  # this frequency centre

        mask = pu.image.make_filter_gaussian(A.shape, peak=peak,
                                             width=sf_band,
                                             pixel_units=True,
                                             zero_mean=True)
        mask[mask < 0.01] = 0  # cut tails of Gaussian

        # mask = pu.image.make_filter_log_cosine(A.shape, peak=peak,
        #                                        pixel_units=True,
        #                                        zero_mean=True)

        res = A * mask
        # take mean of nonzero region:
        mean = np.mean(res[res > 0])
        freq_amp.append(mean)

        inc += sf_step  # in pixel unit of radial increments (cpi).
        if peak >= A.shape[0]/2:
            # if freq greater than or equal to nyquist:
            finished = 1
    return freq, freq_amp


def _do_angular_scan(A, ori_band, ori_step, symmetric):
    """
    Internal function to scan through orientations by applying
    an angular Gaussian filter.

    This one returns angular values corresponding directly
    to the fourier amplitude spectrum. That is, it scans from
    "right" (0) counterclockwise, averaging over a wedge shape.

    The returned angles run from 0--2pi, where 0 or pi is *vertical* modulation
    (i.e. *horizontal* "edges" formed by sinusoids), pi/2 or 3pi/2 are
    *horizontal* modulations (i.e. *vertical* "edges" formed by sinusoids)
    and e.g. pi/4 is a "/" modulation (i.e. "\" edge).

    """

    # scan through:
    finished = 0
    inc = 0
    ang = []
    ang_amp = []

    if symmetric is True:
        # bowtie, running 0-180
        end_angle = 180
    elif symmetric is False:
        end_angle = 360

    while finished == 0:
        peak = 0 + inc
        ang.append(peak)

        mask = pu.image.make_filter_orientation_gaussian(A.shape,
                                                         peak=peak,
                                                         width=ori_band,
                                                         symmetric=symmetric,
                                                         pixel_units=True,
                                                         zero_mean=True)

        mask[mask < 0.01] = 0

        res = A * mask
        # take mean of nonzero region:
        mean = np.mean(res[res > 0])
        ang_amp.append(mean)
        inc += ori_step
        if peak >= end_angle:
            finished = 1

    # convert deg to radians:
    ang = np.array(ang, dtype=np.float)
    ang /= (180 / np.pi)
    return ang, ang_amp


def spectral_analysis_plot(d, angular_mode="direction"):
    """
    Plot the power spectrum as a function of spatial frequency and orientation.

    Args:
        d: the dictionary returned by spectral_analysis.

    Returns:
        None (outputs a plot)
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(np.log2(d['freq']), np.log(d['freq_amp']), 'k-')
    plt.xlabel('Log2 Frequency Band (Octaves)')
    plt.ylabel('Log Mean Amplitude')

    if angular_mode == "orientation":
        # correct axes:
        d['ang'] = d['ang'] * 2

    plt.subplot(1, 2, 2, projection="polar")
    plt.plot(d['ang'], d['ang_amp'], 'k-')
    plt.xlabel('Angle')
    plt.ylabel('Mean Amplitude')
    plt.show()
