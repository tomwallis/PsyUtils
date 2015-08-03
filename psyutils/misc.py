# miscellaneous functions
# -*- coding: utf-8 -*-

import numpy as _np
import os as _os
import psyutils as _pu
import itertools as _it
import pandas as _pd


def fixation_cross():
    """Return a 256 square numpy array containing a rendering of the
    fixation cross recommended in Thaler et al for low dispersion and
    microsaccade rate. You could rescale this to the appropriate size (outer
    ring should be 0.6 dva in diameter and inner ring 0.2 dva).

    Example:
        Our stimulus display has 40 pixels per degree of visual angle::
        from skimage import transform
        sz = round(40 * 0.6)
        fixation_cross = transform.resize(pu.misc.fixation_cross(), (sz,sz))

    Reference:
        Thaler, L., Schütz, A. C., Goodale, M. A., & Gegenfurtner, K. R. (2013)
        What is the best fixation target? The effect of target shape on
        stability of fixational eye movements. Vision Research, 76(C), 31–42.
    """

    outer_rad = 128
    inner_rad = int((0.2 / 0.6)*outer_rad)  # inner is 0.2

    def _draw_oval(radius):
        im = _np.ones((radius*2, radius*2))
        x = _np.linspace(-radius, radius, num=radius*2)
        xx, yy = _np.meshgrid(x, x)
        rad_dist = (xx**2 + yy**2)**0.5
        im[rad_dist <= radius] = 0
        return(im)

    im = _draw_oval(outer_rad)
    im[outer_rad - inner_rad:outer_rad + inner_rad, :] = 1
    im[:, outer_rad - inner_rad:outer_rad + inner_rad] = 1
    im[outer_rad-inner_rad:outer_rad+inner_rad,
       outer_rad-inner_rad:outer_rad+inner_rad] = _draw_oval(inner_rad)
    return(im)


def draw_box(size, channel='r', width=4):
    """Make a box of a given size that can be placed into images to highlight
    a  region of interest. The middle of the box is transparent (i.e. alpha 0)
    to show what's in the region of interest.

    Args:
        size (tuple or scalar):
            the size of the box in pixels; either square if a scalar is passed
            or (w, h) from tuple.
        channel (string):
            specify box colour according to colour channel ('r', 'g', 'b')
        width (int):
            width of box lines in pixels.

    Returns:
        a numpy array with shape [size, size, 4].
    """

    if channel == 'r':
        chan = 0
    elif channel == 'g':
        chan = 1
    elif channel == 'b':
        chan = 2
    else:
        raise ValueError("don't know what colour channel to use")

    w, h = _pu.image.parse_size(size)
    box = _np.zeros((h, w, 4))
    box[0:h, 0:width, chan] = 1.
    box[0:h, -width:, chan] = 1.
    box[0:width, 0:w, chan] = 1.
    box[-width:, 0:w, chan] = 1.

    box[0:h, 0:width, 3] = 1.
    box[0:h, -width:, 3] = 1.
    box[0:width, 0:w, 3] = 1.
    box[-width:, 0:w, 3] = 1.

    return(box)


def pix_per_deg(viewing_distance, screen_wh_px, screen_wh_cm,
                average_wh=True):
    """Return the number of pixels per degree of visual angle for a given
    viewing distance of a screen of some resolution and size.

    Note: this assumes a constant viewing distance, so there will be an error
    that increases with eccentricity. For example, at a viewing distance of
    60 cm, something 30 degrees eccentric will be at a distance of 69 cm
    (60 / np.cos(30 * np.pi / 180)), if presented on a flat screen. At that
    viewing distance, the number of pixels per degree will be higher (46
    compared to 40 for the example monitor below) --- i.e. about a 13
    percent size error at 30 degrees.

    Args:
        viewing_distance (float):
            the viewing distance of the screen (screen to subject's eye) in cm.
        screen_wh_px (tuple):
            the width and height of the screen in pixels.
        screen_wh_cm (tuple):
            the width and height of the screen in cm.
        average_wh (boolean, default True):
            if true, computes pix per deg based on the average of the
            width and height.
            If false, returns a tuple (width, height).

    Returns:
        float: the number of pixels per degree of visual angle, assuming a
        constant distance.
        or if average_wh=False, a 2 element numpy array.

    Example::
        dist = 60
        px = (1920, 1080)
        cm = (52, 29)
        pu.misc.pix_per_deg(60, (1920, 1080), (52, 29))
        # gives 40.36 pixels per degree.
    """

    wh_px = _np.array(screen_wh_px)
    wh_cm = _np.array(screen_wh_cm)

    ppd = _np.pi * (wh_px) / _np.arctan(wh_cm / viewing_distance / 2.) / 360.

    if average_wh is True:
        res = ppd.mean()
    elif average_wh is False:
        res = ppd

    return(res)


def expand_grid(data_dict):
    """ A port of R's expand.grid function for use with Pandas dataframes.
    Taken from:
    `http://pandas.pydata.org/pandas-docs/stable/cookbook.html?highlight=expand%20grid`

    Args:
        data_dict:
            a dictionary or ordered dictionary of column names and values.

    Returns:
        A pandas dataframe with all combinations of the values given.


    Examples::
        import psyutils as pu

        print(pu.misc.expand_grid(
            {'height': [60, 70],
             'weight': [100, 140, 180],
             'sex': ['Male', 'Female']})


        from collections import OrderedDict

        entries = OrderedDict([('height', [60, 70]),
                               ('weight', [100, 140, 180]),
                               ('sex', ['Male', 'Female'])])

        print(pu.misc.expand_grid(entries))

    """

    rows = _it.product(*data_dict.values())
    return _pd.DataFrame.from_records(rows, columns=data_dict.keys())


def rad_ang(xy):
    """Return radius and polar angle relative to (0, 0)
    of given x and y coordinates.

    Args:
        xy: a tuple of x and y positions.

    Returns:
        rad, ang: a tuple of radius from centre
            and polar angle (radians):
                right = 0
                top = pi/2
                left = pi (or -pi)
                bottom = -pi/2

    """

    x, y = (xy[0], xy[1])
    # compute radius and angle of patch centre:
    radius = _np.sqrt(x**2 + y**2)
    angle = _np.arctan2(y, x)

    return(radius, angle)


def xy(radius, angle):
    """ returns the x, y coords of a point given a radius
    and angle (in radians).

    Args:
        radius: a float or int specifying the radius
        angle: the polar angle in radians.
                right = 0
                top = pi/2
                left = pi (or -pi)
                bottom = -pi/2

    Returns:
        x, y: a tuple of x and y coordinates.

    """
    x = radius * _np.cos(angle)
    y = radius * _np.sin(angle)

    return(x, y)
