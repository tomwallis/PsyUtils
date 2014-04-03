# run unit tests on image utilities.

import numpy as np
from skimage import img_as_float
import psyutils as pu
from nose.tools import *

############# Testing guess_type ##############


def test_guess_type_1():
    im = np.ndarray((1, 1, 3))
    correct = "RGB"
    assert pu.image.guess_type(im) == correct


def test_guess_type_2():
    im = np.ndarray((1, 1, 4))
    correct = "RGBA"
    assert pu.image.guess_type(im) == correct


def test_guess_type_3():
    im = np.ndarray((1, 1, 2))
    correct = "IA"
    assert pu.image.guess_type(im) == correct


def test_guess_type_4():
    im = np.ndarray((1, 1))
    correct = "I"
    assert pu.image.guess_type(im) == correct


@raises(Exception)
def test_guess_type_5():
    im = np.ndarray((1, 1, 1, 1))
    pu.image.guess_type(im)


@raises(Exception)
def test_guess_type_6():
    im = np.ndarray((1, 1, 5))
    pu.image.guess_type(im)

############# Testing contrast_image ##############


def test_contrast_image_1():
    im = img_as_float(pu.im_data.tiger())
    im = pu.image.contrast_image(im, factor=0.2)
    min_range = round(im.min(), ndigits=2)
    max_range = round(im.max(), ndigits=2)
    mean = round(im.mean(), ndigits=2)
    sd = round(im.std(), ndigits=2)
    assert min_range == 0.21 \
        and max_range == 0.59 \
        and mean == 0.39 \
        and sd == 0.1


def test_contrast_image_2():
    im = img_as_float(pu.im_data.tiger())
    im = pu.image.contrast_image(im, factor=0.2, returns="contrast")
    min_range = round(im.min(), ndigits=2)
    max_range = round(im.max(), ndigits=2)
    mean = round(im.mean(), ndigits=2)
    sd = round(im.std(), ndigits=2)
    assert min_range == -0.1 \
        and max_range == 0.15 \
        and mean == -0.0 \
        and sd == 0.04


def test_contrast_image_3():
    im = img_as_float(pu.im_data.tiger_grey())
    im = pu.image.contrast_image(im, factor=0.1)
    min_range = round(im.min(), ndigits=2)
    max_range = round(im.max(), ndigits=2)
    mean = round(im.mean(), ndigits=2)
    sd = round(im.std(), ndigits=2)
    assert min_range == 0.46 \
        and max_range == 0.56 \
        and mean == 0.52 \
        and sd == 0.02


def test_contrast_image_4():
    im = img_as_float(pu.im_data.tiger_grey())
    im = pu.image.contrast_image(im, factor=0.1, returns="contrast")
    min_range = round(im.min(), ndigits=2)
    max_range = round(im.max(), ndigits=2)
    mean = round(im.mean(), ndigits=2)
    sd = round(im.std(), ndigits=2)
    assert min_range == -0.05 \
        and max_range == 0.05 \
        and mean == 0.0 \
        and sd == 0.02


def test_contrast_image_5():
    im = img_as_float(pu.im_data.tiger_grey())
    im = pu.image.contrast_image(im, sd=0.1)
    min_range = round(im.min(), ndigits=2)
    max_range = round(im.max(), ndigits=2)
    mean = round(im.mean(), ndigits=2)
    sd = round(im.std(), ndigits=2)
    assert min_range == 0.26 \
        and max_range == 0.77 \
        and mean == 0.52 \
        and sd == 0.1

# See
# http://docs.scipy.org/doc/numpy-dev/reference/generated
# /numpy.testing.assert_allclose.html#numpy.testing.assert_allclose
