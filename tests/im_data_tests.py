# tests for image data.

from skimage import img_as_float
import psyutils as pu


def test_tiger_rgb():
    im = img_as_float(pu.im_data.tiger())
    correct_type = "<class 'numpy.ndarray'>"
    string_type = str(type(im))
    min_range = round(im.min(), ndigits=2)
    max_range = round(im.max(), ndigits=2)
    mean = round(im.mean(), ndigits=2)
    sd = round(im.std(), ndigits=2)
    assert string_type == correct_type \
        and im.shape == (427, 640, 3) \
        and min_range == 0.0 \
        and max_range == 1.0 \
        and mean == 0.39 \
        and sd == 0.21


def test_tiger_square():
    im = img_as_float(pu.im_data.tiger_square())
    correct_type = "<class 'numpy.ndarray'>"
    string_type = str(type(im))
    min_range = round(im.min(), ndigits=2)
    max_range = round(im.max(), ndigits=2)
    mean = round(im.mean(), ndigits=2)
    sd = round(im.std(), ndigits=2)
    assert string_type == correct_type \
        and im.shape == (256, 256, 3) \
        and min_range == 0.01 \
        and max_range == 1.0 \
        and mean == 0.45 \
        and sd == 0.23


def test_tiger_grey():
    im = img_as_float(pu.im_data.tiger_grey())
    correct_type = "<class 'numpy.ndarray'>"
    string_type = str(type(im))
    min_range = round(im.min(), ndigits=2)
    max_range = round(im.max(), ndigits=2)
    mean = round(im.mean(), ndigits=2)
    sd = round(im.std(), ndigits=2)
    assert string_type == correct_type \
        and im.shape == (256, 256) \
        and min_range == 0.01 \
        and max_range == 1.0 \
        and mean == 0.52 \
        and sd == 0.19


def test_tiger_rgba():
    im = img_as_float(pu.im_data.tiger_rgba())
    correct_type = "<class 'numpy.ndarray'>"
    string_type = str(type(im))
    min_range = round(im.min(), ndigits=2)
    max_range = round(im.max(), ndigits=2)
    mean = round(im.mean(), ndigits=2)
    sd = round(im.std(), ndigits=2)
    assert string_type == correct_type \
        and im.shape == (427, 640, 4) \
        and min_range == 0.0 \
        and max_range == 1.0 \
        and mean == 0.54 \
        and sd == 0.32


def test_sloan_import():
    sloans = pu.im_data.sloan_letters()
    im = sloans["H"]
    correct_type = "<class 'numpy.ndarray'>"
    string_type = str(type(im))
    min_range = round(im.min(), ndigits=2)
    max_range = round(im.max(), ndigits=2)
    mean = round(im.mean(), ndigits=2)
    sd = round(im.std(), ndigits=2)
    assert string_type == correct_type \
        and im.shape == (256, 256) \
        and min_range == 0.0 \
        and max_range == 1.0 \
        and mean == 0.48 \
        and sd == 0.5
