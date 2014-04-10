# try to keep global imports within the functions...


def guess_type(image):
    """Make an educated guess about an image's dimensions and what they mean.
    Modified from skimage.color.guess_spatial_dimensions.

    If the image has two dimensions, it is assumed to contain intensity values
    (i.e. be a greyscale image).
    If the image has three dimensions and the third is length 3, it is assumed
    to be an RGB or similar tri-colour image
    (third dimension indexes colour channel).
    If the image is 3D with the third dimension	length 2, it is assumed to be
    an IA image
    (I = intensity, A = alpha (transparency)).
    If the image is 3D with the third dimension	length 4, it is assumed to be
    an RGBA image.

    Args:
        image: ndarray
            The input image.

    Returns:
        assumption (string): how to treat the image in the future.
            Returns a string telling future scripts how to treat the image for
            filtering. Can be:
            "I" (intensity), "IA" (intensity, alpha),
            "RGB", "RGBA" (rgb, alpha).

    Raises:
        ValueError: If the image array has less than two or more than
        three dimensions.

    """
    if image.ndim == 2:
        return "I"
    if image.ndim == 3 and image.shape[-1] == 1:
        raise ValueError("You seem to have passed an array with "
                         "a third dimension of length 1. Please drop the 3rd "
                         "dimension using np.squeeze and try again.")
    if image.ndim == 3 and image.shape[-1] == 2:
        return "IA"
    if image.ndim == 3 and image.shape[-1] == 3:
        return "RGB"
    if image.ndim == 3 and image.shape[-1] == 4:
        return "RGBA"
    else:
        raise ValueError("Expected 2D or 3D array, got %iD." % image.ndim)


def contrast_image(image, factor=1.0, sd=None,
                   returns="intensity",
                   img_dims=None, verbose=False):
    """Takes an input image, applies a transform to each colour channel,
    then returns either the image in intensity units or in
    zero-mean (contrast) units. The latter can be useful if you intend to
    multiply the image by any filters in the non-fourier domain.

    The transform can either be a multiplicative change (specified by
    factor) or the user can specify the `sd` argument, in which case each
    colour channel will have this standard deviation. If `sd` is specified
    then `factor` is ignored.

    Note: the image will be converted to float, if it wasn't already,
    and a copy will be created (i.e. this function doesn't overwrite
    the original image you pass in).

    Args:
        image (ndarray): The input image.
        factor (float, optional): the contrast scale factor.
            A multiplicative contrast change factor. Values less
            than one will reduce global contrast, values greater
            than one will increase global contrast.
        sd (float, optional): the desired standard deviation.
            Each colour channel will be set to have this sd. Note
            this means that relative differences between colour
            channels in the original image will be moved, meaning
            that the colour gamut will not be the same. If
            `sd` is specified `factor` is ignored.
        returns (string, optional): which image to return.
            If "intensity" (the default), the image is returned
            in intensity units (the original scale) after contrast scaling.
            If returns = "contrast", the array returned has mean zero
            in each colour channel.
        img_dims (string, optional): specify image type.
            If "None", check image dimensions with guess_type. Otherwise,
            specify either "I", "IA", "RGB", "RGBA".
            See guess_type? for more details.
        verbose (bool, optional): If True, print additional information.

    Returns:
        image: ndarray
            the modified image. Either zero mean (if "returns" == "contrast")
            or with the original mean. Returned as floating point array.

    """
    import numpy as np
    from skimage import img_as_float

    image = img_as_float(np.copy(image))

    if img_dims is None:
        im_type = guess_type(image)
    else:
        im_type = img_dims

    if verbose is True:
        print("contrast_image function assumes image to be " + im_type)
        if sd is None:
            print("Image will change by factor " + str(factor))
        else:
            print("Intensity SD will be set to " + str(sd))

    if im_type is "I":
        channel_means = np.array(image.mean())
        image = image - channel_means

        if sd is None:
            image *= factor
        else:
            image = (image / image.std()) * sd

        if returns is "intensity":
            image = image + channel_means

    elif im_type is "RGB":
        channel_means = np.zeros(3)
        for i in range(0, 3):
            channel_means[i] = image[:, :, i].mean()
            image[:, :, i] = image[:, :, i] - channel_means[i]
            if sd is None:
                image[:, :, i] = image[:, :, i] * factor
            else:
                image[:, :, i] = (image[:, :, i] / image[:, :, i] .std()) * sd
            if returns is "intensity":
                image[:, :, i] = image[:, :, i] + channel_means[i]

    elif im_type is "IA":
        i = 0
        channel_means = image[:, :, i].mean()
        image[:, :, i] = image[:, :, i] - channel_means
        if sd is None:
            image[:, :, i] = image[:, :, i] * factor
        else:
            image[:, :, i] = (image[:, :, i] / image[:, :, i] .std()) * sd

        if returns is "intensity":
            image[:, :, i] = image[:, :, i] + channel_means

    elif im_type is "RGBA":
        channel_means = np.zeros(3)
        for i in range(0, 3):
            channel_means[i] = image[:, :, i].mean()
            image[:, :, i] = image[:, :, i] - channel_means[i]
            if sd is None:
                image[:, :, i] = image[:, :, i] * factor
            else:
                image[:, :, i] = (image[:, :, i] / image[:, :, i] .std()) * sd
            if returns is "intensity":
                image[:, :, i] = image[:, :, i] + channel_means[i]
    else:
        raise ValueError("Not sure what to do with image type " + im_type)

    if returns is "intensity" or "contrast":
        return image
    else:
        raise ValueError("Not sure what to return from " + returns)


# helper function to show an image and report some stats about it:
def show_im(im):
    """ A helper function to show an image (using imshow)
    and report some stats about it.

    Args:
        image: ndarray
            The input image.

    """
    import matplotlib.pyplot

    dims = guess_type(im)
    if dims is "I" or "IA":
        matplotlib.pyplot.imshow(im, cmap=matplotlib.pyplot.cm.gray)
        #print("note that imshow normalises I image for display")
    elif dims is "RGB" or "RGBA":
        matplotlib.pyplot.imshow(im)
    else:
        raise ValueError("Not sure what to do with image type " + dims)
    print("image is of type " + str(type(im)))
    print("image has dimensions " + str(im.shape))
    print("image has range from " + str(round(im.min(), ndigits=2))
          + " to max " + str(round(im.max(), ndigits=2)))
    print("the mean of the image is " + str(round(im.mean(), ndigits=2)))
    print("the SD of the image is " + str(round(im.std(), ndigits=2)))


def put_rect_in_rect(rect_a, rect_b,
                     mid_x=None, mid_y=None):
    """A function to place a rect inside another rect.

    This function will place np.ndarray `a` into np.ndarray
    `b`, centred on the point mid_x, mid_y. Currently only for 2D arrays.

    Args:
        rect_a (np.ndarray):
            the source rectangle.
        rect_b (np.ndarray):
            the destination rectangle.
        mid_x (int, optional):
            the horizontal position to place the centre of rect_a in rect_b.
            Defaults to the middle of rect_b.
        mid_y (int, optional):
            the vertical position to place the centre of rect_a in rect_b.
            Defaults to the middle of rect_b.

    Returns:
        np.ndarray containing the new rectangle.

    """

    new_rect = rect_b.copy()

    if mid_x is None:
        mid_x = int(rect_b.shape[1] / 2)
    if mid_y is None:
        mid_y = int(rect_b.shape[0] / 2)

    rect_a_rad_x = int(rect_a.shape[1]/2)
    rect_a_rad_y = int(rect_a.shape[0]/2)

    x_start = mid_x - rect_a_rad_x - 1
    y_start = mid_y - rect_a_rad_y - 1
    x_end = x_start + rect_a.shape[1]
    y_end = y_start + rect_a.shape[0]

    if x_start < 0 or y_start < 0:
        raise ValueError("Rect_a falls outside rect_b! " +
                         "x_start is " + str(x_start) +
                         " , y_start is " + str(y_start))
    if x_end > rect_b.shape[1] or y_end > rect_b.shape[0]:
        raise ValueError("Rect_a falls outside rect_b!" +
                         "x_end is " + str(x_end) +
                         " , y_end is " + str(y_end))

    new_rect[y_start:y_end, x_start:x_end] = rect_a

    return(new_rect)
