# tests for miscellaneous functions.

import numpy as np
#from skimage import img_as_float, img_as_ubyte, img_as_uint, io
import psyutils as pu
from nose.tools import *
#import os


def test_pix_per_deg():
    res = pu.misc.pix_per_deg(60, (1920, 1080), (52, 29))
    desired = 40.361110682401332
    np.testing.assert_allclose(res, desired)


def test_pix_per_deg_2():
    res = pu.misc.pix_per_deg(60, (1920, 1080), (52, 29), average_wh=False)
    desired = np.array([40.97539747, 39.7468239])
    np.testing.assert_allclose(res, desired)
