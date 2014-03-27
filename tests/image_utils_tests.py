# run unit tests on image utilities.

import numpy as np
#from skimage import img_as_float
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


# See
# http://docs.scipy.org/doc/numpy-dev/reference/generated/numpy.testing.assert_allclose.html#numpy.testing.assert_allclose