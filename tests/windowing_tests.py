# tests for windowing functions.

import numpy as np
# from skimage import img_as_float
import psyutils as pu
import os

wd = os.getcwd()

if wd[-5:] == 'tests':
    wd = wd[0:-5]

test_data_dir = os.path.join(wd, 'tests', 'test_data')

def test_cos_1d_1():
    true = np.load(file=os.path.join(test_data_dir, 'cos_1d_test.npy'))
    test = pu.image.cos_win_1d(size=20, ramp=5)
    assert np.allclose(test, true)


def test_gauss_2d():
    true = np.load(file=os.path.join(test_data_dir, 'gauss_2d_test.npy'))
    test = pu.image.gaussian_2d(im_x=64, sd_x=16, sd_y=8, ori=45)
    assert np.allclose(test, true)


def test_cos_2d_1():
    true = np.load(file=os.path.join(test_data_dir, 'cos_2d_test_1.npy'))
    test = pu.image.cos_win_2d(size=64, ramp=0.2)
    assert np.allclose(test, true)


def test_cos_2d_2():
    true = np.load(file=os.path.join(test_data_dir, 'cos_2d_test_2.npy'))
    test = pu.image.cos_win_2d(size=16, ramp=4, ramp_type='pixels')
    assert np.allclose(test, true)
