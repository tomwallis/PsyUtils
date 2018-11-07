# tests for miscellaneous functions.

import numpy as np
#from skimage import img_as_float, img_as_ubyte, img_as_uint, io
import psyutils as pu
from nose.tools import *
#import os
from pandas.util.testing import assert_frame_equal
import pandas as pd


def test_pix_per_deg():
    res = pu.misc.pix_per_deg(60, (1920, 1080), (52, 29))
    desired = 40.361110682401332
    np.testing.assert_allclose(res, desired)


def test_pix_per_deg_2():
    res = pu.misc.pix_per_deg(60, (1920, 1080), (52, 29), average_wh=False)
    desired = np.array([40.97539747, 39.7468239])
    np.testing.assert_allclose(res, desired)


def test_expand_grid():
    df = pu.misc.expand_grid({'height': [60, 70],
                                 'weight': [100, 140, 180]})

    desired = pd.DataFrame({'height': [60, 60, 60, 70, 70, 70],
                            'weight': [100, 140, 180, 100, 140, 180]})

    # Because dicts don't follow a consistent order (specified at runtime to
    # reduce memory requirements), we need to sort these and their indices
    # to ensure the assert will pass consistently.

    # same column order
    df.sort_index(axis=1, inplace=True)
    desired.sort_index(axis=1, inplace=True)

    # same row order:
    df.sort_values(by='height', inplace=True)
    desired.sort_values(by='height', inplace=True)

    df.reset_index(drop=True, inplace=True)
    desired.reset_index(drop=True, inplace=True)

    # test
    assert_frame_equal(df, desired, check_names=False)

# doesn't yet work in python 3.3 run_tests.sh for some reason...
# def test_expand_grid():
#     from collections import OrderedDict
#     entries = OrderedDict([('height', [60, 70]),
#                            ('weight', [100, 140, 180])])

#     df = pu.misc.expand_grid(entries)
#     print(df)
#     desired = pd.DataFrame({'height': [60, 60, 60, 70, 70, 70],
#                             'weight': [100, 140, 180, 100, 140, 180]})
#     print(desired)
#     try:
#         assert_frame_equal(df, desired, check_names=False)
#         return True
#     except AssertionError:
#         return False
