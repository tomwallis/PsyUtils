# run unit tests on filters.

import numpy as np
import psyutils as pu
from nose.tools import *
import os

test_data_dir = os.path.join('tests', 'test_data')

#### Test filter creation (to be multiplied in fourier domain) ####

# def test_lowpass():
