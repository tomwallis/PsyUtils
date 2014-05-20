# -*- coding: utf-8 -*-

"""axes.py

A module containing functions that generate and operate on different axes
types. David Janssen wrote it.

"""

import numpy as np

#-----------------------------------------------------------------------------#
#                                 Local Utility                               #
#-----------------------------------------------------------------------------#


def unpack_container(*args):
    """Unpack a container

    If container contains a single entry, which is a container as well, return
    the nested container. In all other cases, return the original container.

    Suggested use:
    def foo(*args):
        tuple(unpack_container(args))

    foo(1, 2, 3) == foo((1, 2, 3)) == foo(((1, 2, 3)))
    t

    """
    if len(args) == 1 and hasattr(args[0], 'count'):
        return unpack_container(*args[0])
    else:
        return args


def parse_axes_limits(*args):
    """Return parsed axes limits

    Axes-limits are defined as:
    [xmin xmax ymin ymax]

    Arguments to this function can be provided either as tuple, or as separate
    arguments, they are treated equivalently. Furthermore, the input can be
    either a single number, a pair of number, or a set of four numbers.

    Axes limits can be:
      - n -> (-n, n, -n, n)
      - (a, b) -> (a, b, a, b)
      - (a, b, c, d) -> (a, b, c, d)

    If axes-limits turn out to be unparseable, an assertion error will be
    thrown.

    examples:
    >>> parse_axes_limits(1)
    (-1, 1, -1, 1)
    >>> parse_axes_limits(-1, 1)
    (-1, 1, -1, 1)
    >>> parse_axes_limits((-1, 1))
    (-1, 1, -1, 1)
    >>> parse_axes_limits(-1, 1, -1, 1)
    (-1, 1, -1, 1)
    >>> parse_axes_limits((-1, 1, -1, 1))
    (-1, 1, -1, 1)

    """
    # Unpack the args
    lim = unpack_container(args)

    #NOTE: This feels hacky...
    if isinstance(lim[0], np.ndarray):
        lim = tuple(lim[0])

    # Construct the lim-tuple
    if len(lim) == 1:
        lim = (-args[0], args[0])
    if len(lim) == 2:
        lim = lim + lim

    # Sanity check the lim tuple
    if (lim[0] >= lim[1] or lim[2] >= lim[3]):
        raise ValueError('Incorrect limit: {0}'.format(lim))
    return lim


def parse_size(*args):
    """Return a size tuple

    If a size is specified as a single number, n, return (n, n).
    If a size is specified as a two tuple, (n, m), return (n, m).
    Otherwise throw an error.

    """
    sz = unpack_container(args)
    if len(sz) == 1:
        sz = sz + sz

    # Sanity check the size tuple
    if (sz[0] <= 0 or sz[1] <= 0):
        raise ValueError('Size must be a positive')
    if len(sz) != 2:
        raise ValueError('Size must be of length 1 or 2')
    return sz


#-----------------------------------------------------------------------------#
#                             Axes transformations                            #
#-----------------------------------------------------------------------------#


def convert_to_polar(x, y):
    """Return a polar representation of x, y

    Returns a tuple containing: (radial, angular) component

    """
    return np.sqrt(x**2 + y**2), np.arctan2(y, x)


def convert_to_log(x, zero_case=None):
    """Return the log of a matrix.

    Silences the divide-by-zero warning before calculating the log. If
    zero_case is specified, all the -inf's in the resulting logged matrix are
    replaced with that value.

    """
    old = np.geterr()['divide']
    np.seterr(divide='ignore')
    out = np.log(x)
    if zero_case is not None:
        out = np.where(out == -np.inf, zero_case, out)
    np.seterr(divide=old)
    return out


def convert_to_exp(x):
    """Return the exp of a matrix

    Currently a *direct* remap of np.exp

    """
    return np.exp(x)


def convert_to_angular_distance(a):
    """Convert a polar axis to angular distance"""
    return np.abs(a)


def rotate_cartesian(x, y, angle=0):
    """Return x and y, both rotated by 'angle'"""
    x2 = np.sin(angle) * y + np.cos(angle) * x
    y2 = np.cos(angle) * y - np.sin(angle) * x
    return x2, y2


def rotate_angular(a, angle=0):
    """Return the provided angular-axes rotated by 'angle'

    Assumes that the angular-axes are specified in radians.

    Angles always specified as between -pi and pi
    """
    a = a + angle
    a = np.where(a > np.pi, a - 2 * np.pi, a)
    return a

#-----------------------------------------------------------------------------#
#                               Axes Generation                               #
#-----------------------------------------------------------------------------#


def cart_axes(size, axes_limits=1, angle=0):
    """Return two numpy arrays of x and y coordinates

    parameters:
      size - size of the resulting matrix in w, h
      axes_limits - (min_x, max_x, min_y, max_y) of the grid

    Check the documentation on parse_size and parse_axes_limits to see how size
    and lims can be specified.

    """
    w, h = parse_size(size)
    l, r, t, b = parse_axes_limits(axes_limits)
    x, y = np.meshgrid(np.linspace(l, r, w),
                       np.linspace(b, t, h))
    return rotate_cartesian(x, y, angle)


def polar_axes(size, axes_limits=1, angle=0):
    """Return two numpy arrays of radial and angular coordinates

    parameters:
      size - size of the resulting matrix in w, h
      axes_limits - (min_x, max_x, min_y, max_y) of the grid



    Check the documentation on parse_size and parse_axes_limits to see how size
    and lims can be specified.

    """
    x, y = cart_axes(size, axes_limits, angle)
    return convert_to_polar(x, y)


def loglog_cart_axes(size, axes_limits=(0, 1), angle=0, zero_case=None):
    """Return a loglog cartesian x and y axis

    All the logs of 0 are replaced by zero_case if its specified, (otherwise
    they default to -inf)

    """
    axes_limits = np.exp(parse_axes_limits(axes_limits))
    x, y = cart_axes(size, axes_limits)
    return convert_to_log(x, zero_case), convert_to_log(y, zero_case)


def semilogx_cart_axes(size, axes_limits=(0, 1, -1, 1),
                       angle=0, zero_case=None):
    """Return a cartesian log x and linear y axis

    All the logs of 0 are replaced by zero_case if its specified, (otherwise
    they default to -inf)

    """
    axes_limits = parse_axes_limits(axes_limits)
    axes_limits = tuple(np.exp(axes_limits[0:2])) + axes_limits[2:4]
    x, y = cart_axes(size, axes_limits, angle)
    return convert_to_log(x, zero_case), y


def semilogy_cart_axes(size, axes_limits=(-1, 1, 0, 1),
                       angle=0, zero_case=None):
    """Return a cartesian linear x and log y axis

    All the logs of 0 are replaced by zero_case if its specified, (otherwise
    they default to -inf)

    """
    axes_limits = parse_axes_limits(axes_limits)
    axes_limits = axes_limits[0:2] + tuple(np.exp(axes_limits[2:4]))
    x, y = cart_axes(size, axes_limits, angle)
    return x, convert_to_log(y, zero_case)


def logradial_polar_axes(size, axes_limits=1, angle=0, zero_case=None):
    """Return a polar axes where the radial component has been logged

    All the logs of 0 are replaced by zero_case if its specified, (otherwise
    they default to -inf).

    """
    r, a = polar_axes(size, axes_limits, angle)
    return convert_to_log(r, zero_case), a


def angular_distance_axes(size, axes_limits=1, angle=0):
    """Return an angular distance axis"""
    r, a = polar_axes(size, axes_limits, angle)
    return convert_to_angular_distance(a)
