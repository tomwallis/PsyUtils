"""
Image submodule
---------------

This submodule contains functions for working with images and creating stimuli
for psychophysical experiments.

----------
"""

from ._axes import(unpack_container, parse_axes_limits, parse_size,
                   convert_to_polar, convert_to_log, convert_to_exp,
                   convert_to_angular_distance,
                   rotate_cartesian, rotate_angular,
                   axes_cart, axes_polar, axes_loglog_cart,
                   axes_semilogx_cart, axes_semilogy_cart,
                   axes_logradial_polar, axes_angular_distance)

from ._image_utilities import (guess_type, contrast_image, show_im, save_im,
                               put_rect_in_rect)

from ._windowing import (cos_win_1d, gaussian_2d, cos_win_2d, wedge_win)

from ._misc import (grid_distort, make_filtered_noise, filter_image)

__all__ = ['unpack_container',
           'parse_axes_limits',
           'parse_size',
           'convert_to_polar',
           'convert_to_log',
           'convert_to_exp',
           'convert_to_angular_distance',
           'rotate_cartesian',
           'rotate_angular',
           'axes_cart',
           'axes_polar',
           'axes_loglog_cart',
           'axes_semilogx_cart',
           'axes_semilogy_cart',
           'axes_logradial_polar',
           'axes_angular_distance',
           'guess_type',
           'contrast_image',
           'show_im',
           'save_im',
           'put_rect_in_rect',
           'cos_win_1d',
           'gaussian_2d',
           'cos_win_2d',
           'wedge_win',
           'make_filter',
           'make_filtered_noise',
           'filter_image',
           'grid_distort',
           ]
