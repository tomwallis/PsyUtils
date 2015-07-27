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
                   axes_logradial_polar, axes_angular_distance,
                   axes_limits_in_pixels)

from ._image_utilities import (guess_type, contrast_image, save_im,
                               ia_2_rgba, put_rect_in_rect, cutout_patch,
                               linear_rescale,
                               alpha_blend)

from ._plot import (show_im, plot_orientations)

# from ._gabor_filtering import

from ._windowing import (cos_win_1d, gaussian_2d, cos_win_2d, wedge_win)

from ._filters import (make_filter_generic,
                       make_filter_lowpass,
                       make_filter_highpass,
                       make_filter_gaussian,
                       make_filter_log_gauss,
                       make_filter_log_exp,
                       make_filter_log_cosine,
                       make_filter_alpha_over_f,
                       make_filter_orientation_gaussian)

from ._misc import (grid_distort, make_filtered_noise, filter_image,
                    diff_ims_error)

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
           'axes_limits_in_pixels',
           'guess_type',
           'contrast_image',
           'save_im',
           'ia_2_rgba',
           'put_rect_in_rect',
           'cutout_patch',
           'linear_rescale',
           'alpha_blend',
           'show_im',
           'plot_orientations',
           'make_filter_generic',
           'make_filter_lowpass',
           'make_filter_highpass',
           'make_filter_gaussian',
           'make_filter_log_gauss',
           'make_filter_log_exp',
           'make_filter_log_cosine',
           'make_filter_alpha_over_f',
           'make_filter_orientation_gaussian',
           'cos_win_1d',
           'gaussian_2d',
           'cos_win_2d',
           'wedge_win',
           'make_filter',
           'grid_distort',
           'make_filtered_noise',
           'filter_image',
           'diff_ims_error'
           ]
