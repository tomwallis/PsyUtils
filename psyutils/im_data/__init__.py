"""
im_data
-----------

This submodule contains some images that we will use for demo and test purposes.

Much of the code in here was adapted from skimage.data.__init__.py

Tiger image Copyright flickr user Phalinn Ooi, released under a CC by 2.0 license.
This image can be freely shared and adapted.
License: https://creativecommons.org/licenses/by/2.0/legalcode

----------
"""

import os as _os

from skimage.io import imread as _imread

_this_dir = _os.path.abspath(_os.path.dirname(__file__))

__all__ = ['tiger',
           'tiger_rgba',
           'tiger_square',
           'tiger_grey']


def _load(f):
    """Load an image file located in the data directory.

    Args:
        f (string): File name.

    Returns:
        img : ndarray
        An image loaded from the psyutils.im_data directory.

    """
    return _imread(_os.path.join(_this_dir, f))


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
    """Load square version of tiger image (RGB), 256 by 256.

    Args:
    none

    Returns:
        img : ndarray

    """
    return _load("tiger_square.png")


def tiger_grey():
    """Grey, square tiger image (RGB), 256 by 256.

    Args:
    none

    Returns:
        img : ndarray

    """
    return _load("tiger_grey.png")
