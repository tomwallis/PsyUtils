# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# #Ipython Notebook for testing and demoing the psyutils package
# 
# In this notebook I will run through several examples of using the psyutils package.

# <markdowncell>

# ## image subpackage

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import img_as_float
#import sys, os


# <markdowncell>

# What's in the `psyutils` package once we load it?

# <codecell>

import psyutils as pu
from psyutils.image import show_im

dir(pu)

# <codecell>

test_im = img_as_float(pu.im_data.tiger_square())
show_im(test_im)

# <codecell>

# lower the contrast of test image, display:
mod_im = pu.image.contrast_image(test_im, factor=0.2, verbose=True)
show_im(mod_im)

# <codecell>

# do the same for a greyscale image:
from skimage.color import rgb2gray

grey_im = rgb2gray(np.copy(test_im))
show_im(grey_im)

# <codecell>

# lower the contrast of greyscale image, display:
grey_mod = pu.image.contrast_image(grey_im, factor=0.2, verbose=True)
show_im(grey_mod)

# <codecell>

# <codecell>
