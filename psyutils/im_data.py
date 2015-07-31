"""
im_data
-----------

This submodule contains some images that we will use for demo
and test purposes.

Much of the code in here was adapted from skimage.data.__init__.py

Tiger image Copyright flickr user Phalinn Ooi, released under a
CC by 2.0 license.
This image can be freely shared and adapted.
License: https://creativecommons.org/licenses/by/2.0/legalcode

----------
"""

import os as _os
import numpy as _np
from skimage import img_as_float

_this_dir = _os.path.abspath(_os.path.dirname(__file__))
_im_dir = 'data'  # where are the images stored?

__all__ = ['tiger',
           'tiger_rgba',
           'tiger_square',
           'tiger_grey',
           'sloan_letters',
           'orientation_test']


def _load(f, **kwargs):
    """Load an image file located in the data directory.

    Args:
        f (string): File name.

    Returns:
        img : ndarray
        An image loaded from the psyutils.im_data directory.

    """
    from skimage.io import imread as _imread

    return _imread(_os.path.join(_this_dir, _im_dir, f), **kwargs)


def tiger():
    """Load the Tiger image (RGB file).

    Args:
        none

    Returns:
        img : ndarray

    """
    return _load("tiger_rgb.png")


def tiger_rgba():
    """Load the tiger image with alpha channel (RGBA file).

    Args:
        none

    Returns:
        img : ndarray

    """
    return _load("tiger_rgba.png")


def tiger_square():
    """Load square version of tiger image (RGB), 256 by 256 by 3.

    Args:
        none

    Returns:
        img : ndarray

    """
    return _load("tiger_square.png")


def tiger_grey():
    """Grey, square tiger image (I), 256 by 256.

    Args:
        none

    Returns:
        img : ndarray

    """
    return _np.load(_os.path.join(_this_dir, _im_dir, "tiger_grey.npy"))


def orientation_test():
    """Grey image containing orientation test patterns.

    Args:
        none

    Returns:
        img : ndarray

    """
    return _load("orientations_test.png", as_grey=True)


def orientation_test_2():
    """Grey image containing orientation test patterns.

    Args:
        none

    Returns:
        img : ndarray

    """
    return _load("orientations_test_2.png", as_grey=True)


def sloan_letters():
    """Returns the ten sloan letters as numpy arrays in a dictionary.
    The sloan letters consist of 10 letter characters:
    C, D, H, K, N, O, R, S, V, Z. Each are 256 by 256 pixels.
    The dict can be accessed like this:
    sloans["C"] will return the np.ndarray containing the letter C.

    Args:
        none

    Returns:
        letters (dict): a dictionary whose keys are strings corresponding
        to the letter ID, and whose entries are numpy ndarrays corresponding
        to the sloan letter. The ndarrays are stored as floating point values
        saved by the skimage img_as_float function (bounded 0--1).
    """
    sloans = {}
    letters = ("C", "D", "H", "K", "N", "O", "R", "S", "V", "Z")
    for i in range(0, 10):
        this_file = "sloan_" + str(i) + ".npy"
        this_file = _os.path.join(_this_dir, _im_dir, this_file)
        im = img_as_float(_np.load(this_file))
        sloans[letters[i]] = im
    return(sloans)
