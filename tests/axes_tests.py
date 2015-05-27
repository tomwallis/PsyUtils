# run unit tests on axes.

import numpy as np
import psyutils as pu
from nose.tools import *
import os

test_data_dir = os.path.join('tests', 'test_data')


def test_axes_limits_in_pixels():
    l, r, t, b = pu.image.axes_limits_in_pixels(256)
    assert l == -256/2 \
        and r == 256/2 \
        and t == -256/2 \
        and b == 256/2


def test_axes_limits_in_pixels_2():
    l, r, t, b = pu.image.axes_limits_in_pixels((256, 128))
    assert l == -256/2 \
        and r == 256/2 \
        and t == -128/2 \
        and b == 128/2
