# functions to do a spectral analysis of a greyscale image.
#
# Some parts of this code adapted with thanks from a notebook on
# image statistics by Bart Machilsen and Maarten Demeyer (see
# http://nbviewer.jupyter.org/github/gestaltrevision/python_for_visres/blob/master/Part7/Part7_Image_Statistics.ipynb)

from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
import warnings


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
                      sf_band=5,
                      sf_step=1,
                      ori_band=12,
                      ori_step=4,
                      angular_mode="direction"):
    """Return the mean amplitude for different radii,
    averaging over angle; and for different angles
    averaging over radii. That is, returns spectral
    power as a function of spatial frequency (c/img) and
    orientation.

    Orientation results for the "direction" mode map directly onto
    the fft amplitude space: starting at zero at 3 o'clock ("right")
    and moving counterclockwise through 2 pi. So horizontal energy is
    horizontal modulation (i.e. "vertical" edges), vertical energy is
    *horizontal* edges, etc.

    Warning: the results will critically depend on the bandwidths and
    step sizes and their interactions with image size.


    Args:
        im:
            (2D numpy array). The image to analyse.
        sf_band:
            the bandwidth of the sliding window for frequency,
            in cycles per image (pixels).
        sf_step:
            the step size of the sliding window for frequency,
            in pixels.
        ori_band:
            the bandwidth of the angular sliding window,
            in degrees.
        ori_step:
            the step size of the angular sliding window,
            in degrees.

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

    # warnings.warn("These functions are not well tested! Orientation results \
    #     in particular will depend on the \
    #     orientation bandwidth and step size.")

    # check inputs
    if len(im.shape) > 2:
        raise ValueError("not currently defined for > 2D images")

    # do fft:
    A, P, F = spectral_analysis_fft(im)

    # not using psyutils internal axes because it's easier to
    # define here with pixel units -- return result as cycles per image.
    # get centred grid axes:
    x, y = np.meshgrid(range(A.shape[1]), range(A.shape[0]))
    x = x - np.max(x)/2
    y = y - np.max(y)/2

    freq, freq_amp = _do_radial_scan(x, y, A, sf_band, sf_step)

    if angular_mode == "direction":
        ang, ang_amp = _do_angular_scan_1(x, y, A, ori_band, ori_step)
    elif angular_mode == "orientation":
        raise ValueError("Not yet implemented, sorry!")
        # ang, ang_amp = _do_angular_scan_2(x, y, A, ori_band)
    else:
        raise ValueError("unknown angular mode")

    return {'freq': freq,
            'freq_amp': freq_amp,
            'ang': ang,
            'ang_amp': ang_amp}


def _do_radial_scan(x, y, A, sf_band, sf_step):
    """
    Internal function for computing the radial part of the spectral analysis.
    """
    # Compute distances from center
    radius = np.hypot(x, y)

    finished = 0
    inc = 0
    freq = []
    freq_amp = []
    while finished == 0:
        start = 0 + inc
        end = start + sf_band
        mid_freq = np.mean([start, end])
        freq.append(mid_freq)  # this frequency centre
        mask = np.zeros(A.shape, dtype=bool)
        mask[(radius <= end) * (radius >= start)] = True
        # Do not include the zero-th frequency (overall luminance)
        mask[x.shape[0]//2, y.shape[0]//2] = False

        # this average amplitude:
        freq_amp.append(np.mean(A[mask].flatten())/A.size)
        inc += sf_step  # in pixel unit of radius increments.
        if mid_freq >= x.shape[0]/2:
            # if middle freq greater than or equal to nyquist:
            finished = 1
    return freq, freq_amp


def _do_angular_scan_1(x, y, A, ori_band, ori_step):
    """
    Internal function to scan through orientations.

    This one returns angular values corresponding directly
    to the fourier amplitude spectrum. That is, it scans from
    "right" (0) counterclockwise, averaging over a wedge shape.

    The returned angles run from 0--2pi, where 0 or pi is *vertical* modulation
    (i.e. *horizontal* "edges" formed by sinusoids), pi/2 or 3pi/2 are
    *horizontal* modulations (i.e. *vertical* "edges" formed by sinusoids)
    and e.g. pi/4 is a "/" modulation (i.e. "\" edge).

    """
    # angle goes from 0 through 2*pi, counterclockwise; zero is "right":
    angle = np.arctan2(y, -x)
    angle += np.pi

    ori_band *= (np.pi / 180)  # convert deg to rad
    ori_step *= (np.pi / 180)

    # scan through:
    finished = 0
    inc = 0
    ang = []
    ang_amp = []
    while finished == 0:
        mid_angle = 0 + inc
        lower_angle = mid_angle - ori_band / 2
        upper_angle = mid_angle + ori_band / 2
        ang.append(mid_angle)

        # create a symmetric wedge (average over edge phases):
        mask = np.zeros(A.shape, dtype=bool)

        # hacky way to get angles of wedge right. I'm sure there's a
        # more sensible and easier way.
        if lower_angle < 0:
            lower_angle = np.mod(lower_angle, 2 * np.pi)  # wrap
            mask[(angle >= lower_angle) *
                 (angle <= (2 * np.pi))] = True
            mask[(angle <= upper_angle) *
                 (angle >= 0)] = True
        elif upper_angle > (2 * np.pi):
            upper_angle = np.mod(upper_angle, 2 * np.pi)  # wrap
            mask[(angle >= 0) *
                 (angle <= upper_angle)] = True
            mask[(angle <= 2 * np.pi) *
                 (angle >= lower_angle)] = True
        else:
            mask[(angle >= lower_angle) * (angle <= upper_angle)] = True

        # Do not include the zero-th frequency (overall luminance)
        mask[x.shape[0]//2, y.shape[0]//2] = False

        # this average amplitude:
        ang_amp.append(np.mean(A[mask].flatten())/A.size)
        inc += ori_step
        if mid_angle >= (2 * np.pi):
            finished = 1
    return ang, ang_amp


# def _do_angular_scan_2(x, y, A, ori_band, ori_step):
#     """
#     Internal function to scan through orientations.

#     This one returns "orientation" values by averaging with
#     "bowtie-shaped" regions (that is, edges of opposite polarities are
#     averaged over).

#     The returned angles run from 0--2pi, where

#     """
#     # angle goes from -pi through pi, clockwise; zero is "right":
#     angle = np.arctan2(y, x)

#     # gives "bowtie" wrap, running 0-->pi clockwise on top,
#     # 0 --> pi clockwise on bottom. i.e. pi = left / right in
#     # fourier amplitude space (vertical edges in image domain)
#     # and pi/2 gives vertical in fourier spectrum (horizontal
#     # edges in image space).
#     angle[angle < 0] += np.pi

#     # angles -----------------------------------
#     ori_band *= (np.pi / 180)  # convert deg to rad
#     ori_step *= (np.pi / 180)

#     # scan through:
#     finished = 0
#     inc = 0
#     ang = []
#     ang_amp = []
#     while finished == 0:
#         mid_angle = 0 + inc
#         lower_angle = mid_angle - ori_band / 2
#         upper_angle = mid_angle + ori_band / 2
#         ang.append(mid_angle)
#         print(mid_angle, lower_angle, upper_angle, inc)

#         # create a symmetric wedge (average over edge phases):
#         mask = np.zeros(A.shape, dtype=bool)

#         # hacky way to get angles of bowtie right. I'm sure there's a
#         # more sensible and easier way.
#         if lower_angle < 0:
#             lower_angle += np.pi
#             mask[(angle >= lower_angle) *
#                  (angle <= (mid_angle + np.pi))] = True
#             mask[(angle <= upper_angle) *
#                  (angle >= mid_angle)] = True
#         elif upper_angle > np.pi:
#             upper_angle -= np.pi
#             mask[(angle >= (mid_angle - np.pi)) *
#                  (angle <= upper_angle)] = True
#             mask[(angle <= mid_angle) *
#                  (angle >= lower_angle)] = True
#         else:
#             mask[(angle >= lower_angle) * (angle <= upper_angle)] = True

#         # Do not include the zero-th frequency (overall luminance)
#         mask[x.shape[0]//2, y.shape[0]//2] = False

#         # this average amplitude:
#         ang_amp.append(np.mean(A[mask].flatten())/A.size)
#         inc += ori_step
#         if mid_angle >= np.pi:
#             finished = 1
#     return ang, ang_amp


def spectral_analysis_plot(d):
    """
    Plot the power spectrum as a function of spatial frequency and orientation.

    Args:
        d: the dictionary returned by spectral_analysis.

    Returns:
        None (outputs a plot)
    """
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(np.log(d['freq']), np.log(d['freq_amp']), 'k-')
    plt.xlabel('Log Frequency Band')
    plt.ylabel('Log Mean Amplitude')

    plt.subplot(1, 2, 2, projection="polar")
    plt.plot(d['ang'], d['ang_amp'], 'k-')
    plt.xlabel('Angle')
    plt.ylabel('Mean Amplitude')
    plt.show()
