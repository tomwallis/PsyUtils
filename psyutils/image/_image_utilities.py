import math as math
import warnings as warnings
import numpy as np
from skimage import img_as_float

def guess_dims(image):
    """Make an educated guess about an image's dimensions and what they mean.
    Modified from skimage.color.guess_spatial_dimensions.

	If the image has two dimensions, it is assumed to contain intensity values
	(i.e. be a greyscale image).

	If the image has three dimensions and the third is length 3, it is assumed to
	be an RGB or similar tri-colour image (third dimension indexes colour channel).

	If the image is 3D with the third dimension	length 2, it is assumed to be an
	IA image (I = intensity, A = alpha (transparency)).

	If the image is 3D with the third dimension	length 4, it is assumed to be an
	RGBA image.

    Parameters
    ----------
    image : ndarray
        The input image.

    Returns
    -------
    assumption : string
        Returns a string telling future scripts how to treat the image for
        filtering. Can be:
        "greyscale"
        "IA"
        "RGB"
        "RGBA"

    Raises
    ------
    ValueError
        If the image array has less than two or more than three dimensions.
    """
    if image.ndim == 2:
        return "greyscale"
    if image.ndim == 3 and image.shape[-1] == 2:
        return "IA"
    if image.ndim == 3 and image.shape[-1] == 3:
        return "RGB"
    if image.ndim == 3 and image.shape[-1] == 4:
        return "RGBA"
    else:
        raise ValueError("Expected 2D or 3D array, got %iD." % image.ndim)




def contrast_image (image, factor = 1.0, returns = "intensity",
                    img_dims = None, verbose = False):
    """
    Takes an input image, takes a guess at its spatial dimensions
    (see guess_dims), applies a multiplicative factor to each
    colour channel then returns either the image in intensity units or in
    zero-mean (contrast) units. The latter can be useful if you intend to
    multiply the image by any filters in the non-fourier domain.

    Note: the image will be converted to float, if it wasn't already.

    Parameters
    ----------
    image : ndarray
        The input image.
    factor : float
        A multiplicative contrast change factor. Values less than one will reduce
        global contrast; values greater than one will increase global contrast.
    returns: string
        Which image to return. If "intensity" (the default), the image is returned
        in intensity units (the original scale) after contrast scaling.
        If returns = "contrast", the array returned has mean zero in each colour channel.
    img_dims: string
        If "None", check image dimensions with guess_dims. Otherwise, specify either
        "greyscale", "IA", "RGB", "RGBA". See guess_dims? for more details.
    verbose : boolean
        If True, print additional information.

    Returns
    -------
    image : ndarray
        the modified image.

    """

    image = img_as_float(np.copy(image))

    if img_dims is None:
        im_type = guess_dims(image)
    else :
        im_type = img_dims

    if verbose is True :
        print("Image is assumed to be " + im_type)

    if im_type is "greyscale" :
        channel_means = np.array(image.mean())
        image = image - channel_means
        image = image * factor

    elif im_type is "RGB" :
        channel_means = np.zeros(3)
        for i in range(0,3):
            channel_means[i] = image[:,:,i].mean()
            image[:,:,i] = image[:,:,i] - channel_means[i]
            image[:,:,i] = image[:,:,i] * factor

    elif im_type is "IA":
        channel_means = image[:,:,0].mean()
        image[:,:,0] = image[:,:,0] - channel_means
        image[:,:,0] = image[:,:,0] * factor

    elif im_type is "RGBA":
        channel_means = np.zeros(3)
        for i in range(0,3):
            channel_means[i] = image[:,:,i].mean()
            image[:,:,i] = image[:,:,i] - channel_means[i]
            image[:,:,i] = image[:,:,i] * factor
    else :
        raise ValueError("Not sure what to do with image type " + im_type)

    if returns is "intensity" :
        if im_type is "greyscale":
            image = image + channel_means
        else:
            for i in range(0, np.size(channel_means)):
                image[:,:,i] = image[:,:,i] + channel_means[i]
        return(image)

    elif returns is "contrast" :
        return(image)

    else :
        raise ValueError("Not sure what to return from " + returns)


# helper function to show an image and report some stats about it:
def show_im(im):
    """ A helper function to show an image (using imshow)
    and report some stats about it.

    Parameters
    ----------
    image : ndarray
        The input image.
    """
    from matplotlib.pyplot import imshow
    from matplotlib.pyplot import cm

    dims = guess_dims(im)
    if dims is "greyscale" or "IA":
        imshow(im, cmap = cm.gray)
        print("note that imshow normalises this for display")
    elif dims is "RGB" or "RGBA":
        imshow(im)
    else :
        raise ValueError("Not sure what to do with image type " + dims)
    print("image is of type " + str(type(im)))
    print("image has dimensions " + str(im.shape))
    print("image has range from " + str(round(im.min(), ndigits = 2)) + " to max " + str(round(im.max(), ndigits = 2)))
    print("the mean of the image is " + str(round(im.mean(), ndigits = 2)))
    print("the SD of the image is " + str(round(im.std(), ndigits = 2)))
