# functions to do a linear Gabor filterbank and plot results.
# at base, a wrapper on skimage's gabor_kernel.

from __future__ import print_function, division
import numpy as np
import psyutils as pu
from skimage.filters import gabor_kernel
from skimage.color import hsv2rgb
import matplotlib.pyplot as plt
from itertools import product
from scipy.signal import fftconvolve
import pycircstat as circ


#------------------------------------------------------------------------------
#  helper functions (internal)
#------------------------------------------------------------------------------

def _check_inputs(frequencies, orientations, n_orientations):
    frequencies = np.atleast_1d(np.array(frequencies))

    if orientations is None:
        inc = np.pi / n_orientations
        orientations = np.linspace(0, np.pi - inc, num=n_orientations)
        # orientations = np.linspace(-np.pi / 2, np.pi / 2, num=n_orientations)
    else:
        orientations = np.array(orientations)

    orientations = np.atleast_1d(orientations)

    return frequencies, orientations


def _convolve(img, filt):
    """Convolve a filter with an img.
    Wrapper doing padding. Adapted from David Janssen,
    with thanks. """

    padx = filt.shape[1] // 2
    pady = filt.shape[0] // 2
    imat = np.pad(img,
                  ((padx, padx), (pady, pady)),
                  mode='mean')

    out = fftconvolve(imat,
                      filt,
                      'same')
    return out[padx : -padx,
               pady : -pady]


# taken straight from scikit-image source:
def _sigma_prefactor(bandwidth):
    b = bandwidth
    # See http://www.cs.rug.nl/~imaging/simplecell.html
    return 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) * \
        (2.0 ** b + 1) / (2.0 ** b - 1)

#------------------------------------------------------------------------------
#  core functions
#------------------------------------------------------------------------------

def gabor_create(frequency=1, orientation=0,
                 bandwidth=1.4, aspect_ratio=1,
                 n_stds=3, offset=0,
                 ppd=42):

    """Create a Gabor kernel. Wrapper for skimage's
    gabor_kernal function that allows you to specify different
    parameters.

    Args:
        frequency:
            Spatial frequency in cycles per degree (pixels per
            degree must be set appropriately).
        orientation:
            Orientation *orthogonal to the carrier modulation* (that is, orientation
            specifies the "edges" found by the filters, not the orientation of the
            grating) in range 0 (rightward horizontal) to pi (left horizontal).
            Vertical is pi/2. Note this orientation is 90 degrees different to the
            skimage gabor_kernel function.

    """
    ppd = float(ppd)

    # convert parameter values from degrees to pixels:
    freq = frequency / ppd  # gabor filter "freq"

    # convert orientation into "edge" coordinates, running counterclockwise:.
    orientation = (np.pi / 2. - orientation)

    # compute sigma x and y from bandwidth, including aspect ratio:
    sigma_x = _sigma_prefactor(bandwidth) / freq
    sigma_y = aspect_ratio * (_sigma_prefactor(bandwidth) / freq)

    # make a complex kernel at this frequency, orientation
    return gabor_kernel(freq, theta=orientation,
                        sigma_x=sigma_x, sigma_y=sigma_y,
                        n_stds=n_stds, offset=offset)


def gaborbank_vis(frequencies=1, n_orientations=4,
                  orientations=None,
                  ppd=42,
                  **kwargs):

    """Visualise the real component of a bank of Gabor filters.

        **kwargs are passed to gabor_create.
    """

    frequencies, orientations = _check_inputs(frequencies,
                                              orientations,
                                              n_orientations)

    # do plots:
    fig, axes = plt.subplots(nrows=len(frequencies),
                             ncols=len(orientations),
                             figsize=(len(orientations), len(frequencies)))
    plt.gray()

    for ind, (f, ori) in enumerate(product(frequencies, orientations)):
        i, j = np.unravel_index(ind, (len(frequencies), len(orientations)))
        kernel = gabor_create(f, orientation=ori, ppd=ppd, **kwargs)
        axes[i][j].imshow(np.real(kernel), interpolation='none')
        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])


def gaborbank_convolve(img,
                       frequencies=1, n_orientations=4,
                       orientations=None,
                       ppd=42,
                       **kwargs):
    """Convolve a bank of Gabor filters with an image.

    Args:
        img:
            (2D numpy array). The image to convolve.
        frequencies:
            a list, tuple or numpy array containing
            the frequencies to filter (in cycles per degree).
        n_orientations:
            the number of orientation bands to equally
            space across the range of orientations. Ignored
            if `orientations` is not None.
        orientations:
            manually specify some orientations (tuple, list
            or numpy array). 0 is a "horizontal" Gabor (carrier modulates
            vertically); pi/2 is a "vertical" Gabor (carrier modulates)
            horizontally.
        ppd:
            the number of pixels per degree of visual angle
            for the viewing conditions you want. This allows
            you to specify filter frequencies in cycles per degree. If
            you set this value to 1, frequencies are specified in cycles
            per pixel.
            That is, to set filters in cycles per image,
            take freq_cpi / imsize and set ppd = 1.
        **kwargs:
            keyword arguments passed to gabor_create.

    Returns:
        A dictionary containing:
            'res': a 4D numpy array containing the complex-valued convolution
                   results. First two dimensions are the same as img; third
                   dim is frequency, fourth orientation.
            'f': a vector of the frequencies (cycles per deg)
            'theta': a vector of the orientations (radians).

    """

    if img.ndim > 2:
        raise ValueError('Currently only works for one-channel images!')

    frequencies, orientations = _check_inputs(frequencies,
                                              orientations,
                                              n_orientations)

    result = np.empty((img.shape[0], img.shape[1],
                       len(frequencies), len(orientations)),
                       dtype=np.complex128)

    for ind, (f, ori) in enumerate(product(frequencies, orientations)):
        i, j = np.unravel_index(ind, (len(frequencies), len(orientations)))
        kernel = gabor_create(f, orientation=ori, ppd=ppd, **kwargs)
        result[:, :, i, j] = _convolve(img, kernel)

    return {'res': result,
           'f': frequencies,
           'theta': orientations}

#------------------------------------------------------------------------------
#  analysis functions
#------------------------------------------------------------------------------

def gaborbank_mean_orientation(d):
    """ Compute the mean orientation for each point in the image
    by summing energy over spatial frequencies and then computing the
    circular mean of orientations for each point, weighted by the filter
    response.

    Args:
        d: the dict output by gaborbank_convolve.

    Returns:
        image containing interpolated orientations. Possible values
        run 0 to pi radians, increasing counterclockwise (pi/2 is vertical).
    """
    res = d['res']
    theta = d['theta']

    e = res.real**2 + res.imag**2
    e = e.sum(axis=2)  # sum energy over scales.

    # reshape the angles into an image plane for each angle:
    t = np.tile(theta, e.shape[0]*e.shape[1])
    t = np.reshape(t, e.shape)
    # t has the same shape as e, with each orientation along axis 2.

    """compute circular mean, with energy as weights. Axial correction
    is to change direction 0-pi --> orientation 0-2*pi. Correct afterward
    by folding orientations > pi back around.
    """

    out = circ.mean(t, w=e, axis=2, axial_correction=2)
    out[out > np.pi] -= np.pi
    return out


def gaborbank_max_orientation(d):
    """ Compute the "orientation" for each point in the image
    by summing energy over spatial frequencies and then taking the
    orientation of the filter with maximum energy.

    Args:
        d: the dict output by gaborbank_convolve.

    Returns:
        image containing orientations 0--pi where 0/pi is horizontal and
        pi/2 is vertical (increasing counterclockwise).
    """
    res = d['res']
    e = res.real**2 + res.imag**2  # energy
    e = e.sum(axis=2)  # sum energy over scales.

    # which orientation has the max response at each x,y point?
    inds = e.argmax(axis=2)

    return d['theta'][inds]


def gaborbank_phase_angle(d):
    """Compute the phase at each point in the image.

    """
    res = d['res']

    return np.arctan2(res.real, res.imag)


def gaborbank_orientation_variance(d):
    """ Compute the orientation variance. Orientation variance
    can be any value from 0 (no variance, all responses point
    in one direction) to 1 (no dominant direction). Computed
    using pycircstat's mean resultant vector function.

    """
    res = d['res']
    theta = d['theta']

    e = res.real**2 + res.imag**2
    e = e.sum(axis=2)  # sum energy over scales.

    # reshape the angles into an image plane for each angle:
    t = np.tile(theta, e.shape[0]*e.shape[1])
    t = np.reshape(t, e.shape)
    # t has the same shape as e, with each orientation along axis 2.

    """compute orientation variance with energy as weights. Axial correction
    is to change direction 0-pi --> orientation 0-2*pi. Correct afterward
    by folding orientations > pi back around.
    """
    out = circ.resultant_vector_length(t, w=e, axis=2, axial_correction=2)
    return 1 - out


def gaborbank_orientation_vis(d, method='mean', legend=True):
    """ Visualise the orientation for each point in the image.

    Method 'mean' uses the mean resultant vector over
    all orientation filters. Method 'max' takes the orientation
    at each pixel to be that of the filter with maximum energy
    at that pixel.

    Adapted from http://nbviewer.ipython.org/github/gestaltrevision\
    /python_for_visres/blob/master/Part7/Part7_Image_Statistics.ipynb

    Args:
        d: the dict output by gaborbank_convolve.
    """

    res = d['res']
    e = res.real**2 + res.imag**2  # energy
    e = e.sum(axis=3).sum(axis=2)  # sum energy over scales and orientations.

    if method == 'mean':
        ori = gaborbank_mean_orientation(d)
    elif method == 'max':
        ori = gaborbank_max_orientation(d)
    else:
        raise ValueError('Unknown method!')

    # output values range 0--pi; adjust hues accordingly:
    H = ori / np.pi
    S = np.ones_like(H)
    V = (e - e.min()) / e.max()
    HSV = np.dstack((H, S, V))
    RGB = hsv2rgb(HSV)

    if legend is True:
        # Render a hue circle
        sz = int(e.shape[0] * 0.1)
        r, a = pu.image.axes_polar(sz)
        a[a < 0] += np.pi
        a /= np.pi
        # a = (a - a.min()) / a.max()
        a = 1 - a  # not sure why I have to flip this, but
        # otherwise diagonals are reversed.
        mask = (r < 0.9) & (r > 0.3)
        hsv_legend = np.dstack((a,
                                np.ones_like(a, dtype='float'),
                                mask.astype('float')))
        rgb_legend = hsv2rgb(hsv_legend)
        RGB[:sz, :sz, :] = rgb_legend[::-1, ::]

    return RGB


#------------------------------------------------------------------------------
#  TODO functions
#------------------------------------------------------------------------------

# function to return frequency and orientation bandwidths from gabor_create.


