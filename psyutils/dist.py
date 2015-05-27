""" Dist submodule

This submodule contains various probability distributions
that are generally returned as functions. They can then be called
on whatever inputs you like (e.g., axes).

"""


import numpy as _np
from psyutils.image import convert_to_log as _convert_to_log


def log_exponential(peak, width):
    """Return a log-exponential function with a certain peak and width.
    Modified version of equation 1 in Bex (2010), (In) sensitivity to
    spatial distortion in natural scenes. Journal of Vision.

    width is defined as half-bandwith, in octaves.

    """
    A = -1 / (_np.log(2)**2 * width**3)
    log_p = _np.log(peak)
    return lambda x: _np.exp(A * _np.abs(_convert_to_log(x) - log_p)**3)


def one_over_f(alpha):
    """Return a 1/f function"""
    return lambda x: _np.power(x, -alpha)


def log_cosine(peak):
    """Return a log-cosine function"""
    peak = _np.log2(peak)

    def f(x):
        #NOTE: this will give the divide by zero warning, should we generalize
        #_convert_to_log?
        x = _np.log2(x)
        filt = 0.5 * (1 + _np.cos(_np.pi * (x - peak)))
        filt = _np.where((x > (peak + 1)) |
                        (x <= (peak - 1)), 0, filt)
        return filt
    return f


def log_gauss(peak, width):
    return lambda x: _np.exp(-((_np.log2(x) - _np.log2(peak))
                             / (2 * width))**2)


def gauss(peak, width):
    return lambda x: _np.exp(-0.5*((x - peak) / width)**2)


def highpass(peak, include_border=True):
    """NOTE: doc this

    If include_border is true, peak is included in the filtered results,
    otherwise it's excluded

    """
    if include_border:
        return lambda x: _np.where(x <= peak, 0, 1)
    else:
        return lambda x: _np.where(x < peak, 0, 1)


def lowpass(peak, include_border=True):
    if include_border:
        return lambda x: _np.where(x <= peak, 1, 0)
    else:
        return lambda x: _np.where(x < peak, 1, 0)
