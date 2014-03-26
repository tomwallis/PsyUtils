"""Test images.

Much of the code in here was taken from skimage.data.__init__.py

Tiger
---------
Copyright flickr user Phalinn Ooi, released under a CC by 2.0 license.
This image can be freely shared and adapted.
License: https://creativecommons.org/licenses/by/2.0/legalcode


"""

import os as _os

from skimage.io import imread as _imread
from psyutils import _im_data_dir


__all__ = ['tiger',
           'tiger_square',
           'tiger_rgba',
           'tiger_grey']


def _load(f):
    """Load an image file located in the data directory.


    Parameters
    ----------
    :rtype : ndarray
    f : string
        File name.

    Returns
    -------
    img : ndarray
    Image loaded from psyutils.im_data_dir.
    """
    return _imread(_os.path.join(_im_data_dir, f))


def tiger():
    """Tiger image by Phalinn Ooi (flickr).
    Released and used here under a CC by 2.0 license.
    License: https://creativecommons.org/licenses/by/2.0/legalcode
    """
    return _load("tiger_rgb.png")


def tiger_rgba():
    """Tiger image with alpha channel.
    """
    return _load("tiger_rgba.png")


def tiger_square():
    """Square version of tiger image (RGB).
    """
    return _load("tiger_square.png")

def tiger_grey():
    """Grey, square tiger image (RGB).
    """
    return _load("tiger_grey.png")
