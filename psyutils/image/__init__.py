"""
Image submodule
---------------

This submodule contains functions for working with images and creating stimuli
for psychophysical experiments.

----------
"""

from ._image_utilities import (guess_type, contrast_image, show_im)
from ._windowing import (cos_win_1d, gaussian_2d, plot_win_1d)
from ._filters import (make_filter, make_filtered_noise, filter_image)

__all__ = ['guess_type',
           'contrast_image',
           'show_im',
           'cos_win_1d',
           'gaussian_2d',
           'make_filter',
           'make_filtered_noise',
           'filter_image',
           'plot_win_1d']
