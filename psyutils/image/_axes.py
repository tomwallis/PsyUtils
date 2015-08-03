# -*- coding: utf-8 -*-

"""axes.py

A module containing functions that generate and operate on different axes
types. David Janssen wrote them, Tom Wallis tweaked and documented.

"""

from __future__ import print_function, division
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


def axes_cart(size, axes_limits=1, angle=0):
    """Return two numpy arrays of x and y coordinates.
    Check the documentation on parse_size and parse_axes_limits to see how size
    and lims can be specified.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively).
        axes_limits (optional):
            (min_x, max_x, min_y, max_y) of the grid. Defaults to -1, 1.
        angle (optional):
            angle to rotate the axes (in radians).

    Returns:
        x, y (matrices):
            Matrices containing the x and y coordinates of the axes.

    Example:
        # return axes 12 elements square, with a rotation of 0.5*pi:
        x, y = pu.image.axes_cart(size=12, angle=0.5*pi)

        # return axes 12 by 16 elements, scaled from -10 to 10:
        x, y = pu.image.axes_cart(size=(12, 16), axes_limits=10)

    """
    w, h = parse_size(size)
    l, r, t, b = parse_axes_limits(axes_limits)
    x, y = np.meshgrid(np.linspace(l, r, w),
                       np.linspace(b, t, h))
    return rotate_cartesian(x, y, angle)


def axes_polar(size, axes_limits=1, angle=0):
    """Return two numpy arrays of radial and angular coordinates
    Check the documentation on parse_size and parse_axes_limits to see how size
    and lims can be specified.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively).
        axes_limits (optional):
            (min_x, max_x, min_y, max_y) of the grid. Defaults to -1, 1.
        angle (optional):
            angle to rotate the axes (in radians).

    Returns:
        r, a (matrices):
            The r matrix is the radial distance from the centre,
            the a matrix is the angle (from -pi to pi, counterclockwise
            from left (west)).

    Example:
        # return axes 12 elements square, with a rotation of 0.5*pi:
        r, a = pu.image.axes_polar(size=12, angle=0.5*pi)

        # return axes 12 by 16 elements, scaled from -10 to 10:
        r, a = pu.image.axes_polar(size=(12, 16), axes_limits=10)
    """
    x, y = axes_cart(size, axes_limits, angle)
    return convert_to_polar(x, y)


def axes_loglog_cart(size, axes_limits=1, angle=0, zero_case=None):
    """Return a loglog cartesian x and y axis
    Check the documentation on parse_size and parse_axes_limits to see how size
    and lims can be specified.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively).
        axes_limits (optional):
            (min_x, max_x, min_y, max_y) of the grid. These are given in log
            coordinates. So, if you want your axes to run from 1 to 100, you
            should specify limits as (log(1), log(100)). Defaults to -1, 1.
        angle (optional):
            angle to rotate the axes (in radians).
        zero_case (optional):
            All the logs of 0 are replaced by zero_case if its specified,
            (otherwise they default to -inf)

    Returns:
        x, y (matrices):
            Matrices containing the x and y coordinates of the axes, in log
            spacing.

    Example:
        # return axes 12 elements square:
        x, y = pu.image.axes_loglog_cart(size=12)

        # return axes 12 by 16 elements, scaled from exp(-2) to exp(2):
        x, y = pu.image.axes_loglog_cart(size=(12, 16), axes_limits=2)

    """
    axes_limits = np.exp(parse_axes_limits(axes_limits))
    x, y = axes_cart(size, axes_limits)
    return convert_to_log(x, zero_case), convert_to_log(y, zero_case)


def axes_semilogx_cart(size, axes_limits=1,
                       angle=0, zero_case=None):
    """Return a cartesian log x and linear y axis
    Check the documentation on parse_size and parse_axes_limits to see how size
    and lims can be specified.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively).
        axes_limits (optional):
            (min_x, max_x, min_y, max_y) of the grid. These are given in log
            coordinates for the x. So, if you want your x axis to run from
            1 to 100, you should specify limits as (log(1), log(100)).
        angle (optional):
            angle to rotate the axes (in radians).
        zero_case (optional):
            All the logs of 0 are replaced by zero_case if its specified,
            (otherwise they default to -inf)

    Returns:
        x, y (matrices):
            Matrices containing the x and y coordinates of the axes,
            with x log-scaled.

    """
    axes_limits = parse_axes_limits(axes_limits)
    axes_limits = tuple(np.exp(axes_limits[0:2])) + axes_limits[2:4]
    x, y = axes_cart(size, axes_limits, angle)
    return convert_to_log(x, zero_case), y


def axes_semilogy_cart(size, axes_limits=1,
                       angle=0, zero_case=None):
    """Return a cartesian linear x and log y axis
    Check the documentation on parse_size and parse_axes_limits to see how size
    and lims can be specified.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively).
        axes_limits (optional):
            (min_x, max_x, min_y, max_y) of the grid. These are given in log
            coordinates for the y. So, if you want your y axis to run from
            1 to 100, you should specify limits as (log(1), log(100)).
        angle (optional):
            angle to rotate the axes (in radians).
        zero_case (optional):
            All the logs of 0 are replaced by zero_case if its specified,
            (otherwise they default to -inf)

    Returns:
        x, y (matrices):
            Matrices containing the x and y coordinates of the axes,
            with y log-scaled.

    """
    axes_limits = parse_axes_limits(axes_limits)
    axes_limits = axes_limits[0:2] + tuple(np.exp(axes_limits[2:4]))
    x, y = axes_cart(size, axes_limits, angle)
    return x, convert_to_log(y, zero_case)


def axes_logradial_polar(size, axes_limits=1, angle=0, zero_case=None):
    """Return a polar axes where the radial component has been logged
    Check the documentation on parse_size and parse_axes_limits to see how size
    and lims can be specified.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively).
        axes_limits (optional):
            (min_x, max_x, min_y, max_y) of the grid. These are given in log
            coordinates. So, if you want your axes to run from 1 to 100, you
            should specify limits as (log(1), log(100)). Defaults to -1, 1.
        angle (optional):
            angle to rotate the axes (in radians).
        zero_case (optional):
            All the logs of 0 are replaced by zero_case if its specified,
            (otherwise they default to -inf)

    Returns:
        r, a (matrices):
            The r matrix is the log radial distance from the centre,
            the a matrix is the angle (from -pi to pi, counterclockwise
            from left (west)).

    Example:
        # return axes 12 elements square, with a rotation of 0.5*pi:
        r, a = pu.image.axes_logradial_polar(size=12, angle=0.5*pi)

        # return axes 12 by 16 elements, scaled from -10 to 10:
        r, a = pu.image.axes_logradial_polar(size=(12, 16), axes_limits=10)

    """
    # not sure what's going on here...
    # axes_limits = parse_axes_limits(axes_limits)
    # axes_limits = tuple(np.exp(axes_limits[0:2])) + axes_limits[2:4]
    r, a = axes_polar(size, axes_limits, angle)
    return convert_to_log(r, zero_case), a


def axes_angular_distance(size, axes_limits=1, angle=0):
    """Return an angular distance axis.

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively).
        axes_limits (optional):
            (min_x, max_x, min_y, max_y) of the grid. Defaults to -1, 1.
        angle (optional):
            angle to rotate the axes (in radians).

    Returns:
        a (matrix):
            The anglular distance (from 0 to pi).

    Example:
        # return axes 100 elements square, with a rotation of 0.5*pi:
        a = pu.image.axes_angular_distance(size=12, angle=0.5*pi)

    """
    r, a = axes_polar(size, axes_limits, angle)
    return convert_to_angular_distance(a)


def axes_limits_in_pixels(size):
    """A function to define axis limits as pixels, so that the
    returned axis values will correspond to the size of the axes.
    Useful for when you want to have functions with values defined in pixels
    (e.g. filters with peaks defined in cycles per image).

    Args:
        size:
            size of the resulting matrix in w, h (i.e., number of columns and
            rows respectively), or a scalar (in which case the result is square)
    Returns:
        (min_x, max_x, min_y, max_y) of the grid.

    Example:

    """
    w, h = parse_size(size)
    l = -w / 2.
    r = w / 2.
    t = -h / 2.
    b = h / 2.
    return(l, r, t, b)
