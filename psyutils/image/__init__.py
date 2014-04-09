"""
Image submodule
---------------

This submodule contains functions for working with images and creating stimuli
for psychophysical experiments.

----------
"""

from ._image_utilities import (guess_type, contrast_image, show_im)
from ._windowing import (cos_win_1d, gaussian_2d, cos_win_2d, plot_win_1d)
from ._filters import (make_filter, make_filtered_noise, filter_image)
from ._distortions import (bex_distort)
from ._eccentricity import (ecc_linear)

__all__ = ['guess_type',
           'contrast_image',
           'show_im',
           'cos_win_1d',
           'gaussian_2d',
           'cos_win_2d',
           'plot_win_1d',
           'make_filter',
           'make_filtered_noise',
           'filter_image',
           'bex_distort',
           'ecc_linear',
           ]
