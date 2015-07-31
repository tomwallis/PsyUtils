# image plotting functions
import numpy as np
import matplotlib.pyplot as plt
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



def plot_orientations(res, theta, include_legend=True):
    """ Plot the oriented output of a filterbank, with
    intensity weighted by the filter response at each pixel.

    Taken from http://nbviewer.ipython.org/github/gestaltrevision/python_for_visres/blob/master/Part7/Part7_Image_Statistics.ipynb

    res should be the *maximum* response at each pixel (over filter bank)
    theta should be the *orientation* of that maximum response. Will be
    rescaled to 0-1 range.

    That is, these should be 2 dimensional arrays.

    ** TODO:
    - specify required input ranges.
    - improve docs
    - write tests **
    """

    theta = theta.astype(float)
    res = res.astype(float)

    H = (theta-np.min(theta)) / np.max(theta)
    S = np.ones_like(H)
    V = (res-np.min(res)) / np.max(res)
    HSV = np.dstack((H, S, V))
    RGB = hsv2rgb(HSV)

    if include_legend is True:
        # Render a hue circle as legend
        sz = int(res.shape[0] * 0.1)
        x, y = np.meshgrid(range(sz), range(sz))
        rad = np.hypot(x-sz/2, y-sz/2)
        ang = 0.5+(np.arctan2(y-sz/2, x-sz/2)/(2*np.pi))
        mask = (rad < sz/2) & (rad > sz/4)

        hsv_legend = np.dstack((ang,
                                np.ones_like(ang, dtype='float'),
                                mask.astype('float')))
        rgb_legend = hsv2rgb(hsv_legend)
        RGB[:sz, :sz, :] = rgb_legend[::-1, ::]

    return RGB
