# image plotting functions

from __future__ import print_function, division
import numpy as np
import psyutils as pu
from skimage.color import hsv2rgb

# helper function to show an image and report some stats about it:
def show_im(im, n_digits=3, colorbar=False):
    """ A helper function to show an image (using imshow)
    and report some stats about it.

    Args:
        image: ndarray
            The input image.
        n_digits: int
            The number of digits to print for numerical output.

    """
    # matplotlib causes platform-specific import problems.
    # https://github.com/scikit-optimize/scikit-optimize/issues/637#issuecomment-366448262
    try:
       import matplotlib.pyplot as plt
    except ImportError as e:
       if 'Python is not installed as a framework.' in e.message:
         warnings.warn("Warning: this OS has an import error with matplotlib,\
          likely related to backend problem. Search for matplotlib import fix.")

    n_digits = int(n_digits)
    dims = pu.image.guess_type(im)
    if dims is "I":
        plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
    elif dims is "IA":
        # convert to rgba for display:
        rgba = pu.image.ia_2_rgba(im)
        plt.imshow(rgba, interpolation='nearest')
    elif dims is "RGB" or dims is "RGBA":
        plt.imshow(im, interpolation='nearest')
    else:
        raise ValueError("Not sure what to do with image type " + dims)
    print("image is of type " + str(type(im)))
    print("image has data type " + str(im.dtype))
    print("image has dimensions " + str(im.shape))
    print("image has range from " + str(round(im.min(), ndigits=n_digits))
          + " to max " + str(round(im.max(), ndigits=n_digits)))
    print("the mean of the image is " + str(round(im.mean(), ndigits=n_digits)))
    print("the SD of the image is " + str(round(im.std(), ndigits=n_digits)))
    print("the rms contrast (SD / mean) is " +
          str(round(im.std()/im.mean(), ndigits=n_digits)))

    if colorbar == True:
        plt.colorbar();

    plt.axis("off")
