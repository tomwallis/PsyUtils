import numpy as np
from skimage import img_as_float, io, exposure, img_as_uint, color, img_as_ubyte
import matplotlib.pyplot as plt


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

    if returns is "intensity" or returns is "contrast":
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

    dims = guess_type(im)
    if dims is "I":
        plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
    elif dims is "IA":
        # convert to rgba for display:
        rgba = ia_2_rgba(im)
        plt.imshow(rgba, interpolation='nearest')
    elif dims is "RGB" or dims is "RGBA":
        plt.imshow(im, interpolation='nearest')
    else:
        raise ValueError("Not sure what to do with image type " + dims)
    print("image is of type " + str(type(im)))
    print("image has data type " + str(im.dtype))
    print("image has dimensions " + str(im.shape))
    print("image has range from " + str(round(im.min(), ndigits=2))
          + " to max " + str(round(im.max(), ndigits=2)))
    print("the mean of the image is " + str(round(im.mean(), ndigits=2)))
    print("the SD of the image is " + str(round(im.std(), ndigits=2)))
    print("the rms contrast (SD / mean) is " +
          str(round(im.std()/im.mean(), ndigits=2)))


def ia_2_rgba(im):
    """ Convert an MxNx2 image (interpreted as an intensity map plus alpha
    level) to an RGBA image (MxNx4). The channels will be rescaled to the
    range 0--1 and converted to float.

    THIS FUNCTION IS QUICK AND DIRTY AND SHOULD ONLY BE USED TO DISPLAY
    IMAGES, NOT SAVE THEM AT THIS STAGE.
    """

    rgba = np.ndarray((im.shape[0], im.shape[1], 4), dtype=np.float)
    im = img_as_float(im.copy())
    rgb = color.gray2rgb(im[..., 0])
    rgba[..., :3] = rgb
    rgba[..., 3] = im[..., 1]

    rgba = exposure.rescale_intensity(rgba, out_range=(0, 1))

    return(rgba)


def save_im(fname, im, bitdepth=8):
    """
    Takes a numpy array, converts it to either uint8 or uint16, and
    saves it to a .png file by converting it to
    an unsigned integer. This is a wrapper for skimage.io.imsave,
    and calls the freeimage library to allow saving with high
    bit depth (8 or 16). Both scikit-image and Freeimage must be
    installed for this to work. On OSX Freeimage can be installed
    using homebrew.

    If the array passed is MxN, the resulting file (.png) will be
    greyscale. If the file is MxNx3 it will be RGB, if MxNx4 it's
    RGBA.

    Warning: old versions of scikit-image (pre 0.11.0) will mess up the
    order of the colour planes and mirror flip images if bitdepth is 16.

    Args:
        fname (string):
            the filename to save the image to.
        im (ndarray, float):
            a numpy array (either 2D or 3D) to save.
        bitdepth (int):
            either 8 or 16.
    """

    # dims = guess_type(im)
    # im = img_as_float(im)
    # check scale:
    # im = exposure.rescale_intensity(im, out_range='float')

    if bitdepth is 8:
        # convert to uint8:
        im = img_as_ubyte(im)

    elif bitdepth is 16:
        # convert to 16 bit
        im = img_as_uint(im)

        # # to fix bug in current release of scikit-image (issue 1101; closed)
        # if dims == "RGB" or dims == "RGBA":
        #     im = np.fliplr(np.flipud(im))

    io.use_plugin('freeimage')
    io.imsave(fname, im)


def put_rect_in_rect(rect_a, rect_b,
                     mid_x=None, mid_y=None):
    """A function to place rect_a inside rect_b.

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


def linear_rescale(im, maxmin=(-1, 1)):
    """ Linearly rescale an image between the values
    given by the tuple `maxmin`.

    Note this is different to skimage's rescale intensity function,
    which respects the contrast of the values. In contrast, this function
    will stretch the range to maxmin[0], maxmin[1].

    """

    im_std = (im - im.min()) / (im.max() - im.min())
    im_scaled = im_std * (maxmin[1] - maxmin[0]) + maxmin[0]
    return(im_scaled)


def alpha_blend(fg, bg):
    """ Do alpha blending putting foreground `fg` in front of background
    `bg`. Will be converted to floats and rescaled to the range 0, 1 using
    skimage's rescale intensity function, so should respect contrast.

    Input images are assumed to be images whose third dimension is
    4 (i.e. RGBA).

    """

    fg = img_as_float(fg)
    bg = img_as_float(bg)

    fg = exposure.rescale_intensity(fg, out_range=(0, 1))
    bg = exposure.rescale_intensity(bg, out_range=(0, 1))

    # fg = img_as_ubyte(fg)
    # bg = img_as_ubyte(bg)

    fg_rgb = fg[..., :3]
    fg_alpha = fg[..., 3]
    bg_rgb = bg[..., :3]
    bg_alpha = bg[..., 3]

    out_alpha = fg_alpha + bg_alpha * (1. - fg_alpha)

    # # check for any zeros:
    # out_alpha[out_alpha == 0] = 1e-12

    out_rgb = (fg_rgb * fg_alpha[..., None] +
               bg_rgb * (1. - fg_alpha[..., None])) / out_alpha[..., None]

    out = np.zeros_like(bg)
    out[..., :3] = out_rgb
    out[..., 3] = out_alpha
    out = exposure.rescale_intensity(out, out_range=(0, 1))
    return(out)
