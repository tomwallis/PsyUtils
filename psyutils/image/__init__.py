"""
Image submodule
---------------

This submodule contains functions for working with images and creating stimuli for psychophysical experiments.

----------
"""

from ._image_utilities import (guess_type, contrast_image, show_im)

__all__ = ['guess_type',
           'contrast_image',
           'show_im']
