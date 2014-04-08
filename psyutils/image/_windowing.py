# Functions for windowing, including images.


def cos_win_1d(length,
               ramp='default',
               padding=0):
    """Create a vector of length ``length`` containing a 1D cosine window
    where the centre of the window is set to 1 and the ramps go down to
    zero symmetrically on either side (with optional zero padding).

    This can be useful for e.g. adjusting the alpha channel of a stimulus
    over frames so that the stimulus is smoothly ramped on and off.
    Since what's returned is a vector, ``length`` must be an integer
    or convertible to integer.

    This function will do some basic checking to see that your parameters
    make sense but it is not thorough. If parameters are provided as integers
    they will be converted to floats.

    Example:
        Create a vector of length 200 where the value is 1 for the central
        100 samples, on and off ramps of length 45 and 5 samples of zero
        padding at either end::
            window = cos_win_1d(length = 200, ramp = 45, padding = 5)
            pu.image.plot_win_1d(window)

    Args:
        length (int):
            the length of the window function to return.
        ramp (int or float, optional):
            the size of each ramp. Defaults to (length-cent) / 2.
        padding (int, optional):
            the size of zero padding at the edges. Defaults to zero.

    Returns:
        window (float): a vector containing the windowing kernel,
        normalised 0--1.

    """

    import numpy as np

    # check parameters, set.
    length = float(length)
    length_int = int(length)

    if ramp is 'default':
        ramp = length / 4.0
    else:
        ramp = float(ramp)

    padding = int(padding)

    # do a check for params making sense:
    tot = (ramp * 2.0) + (padding * 2.0)

    if tot > length:
        raise ValueError("Your ramping parameters add up to " + str(tot) +
                         " but you " +
                         "asked for length " + str(length))

    y = np.ones((length_int))

    # create the ramps:
    up_ramp = np.sin(np.linspace(0, 1, round(ramp)) * np.pi/2.0)
    down_ramp = np.cos(np.linspace(0, 1, round(ramp)) * np.pi/2.0)

    # place into y:
    if padding is not 0:
        y[padding:(padding+ramp)] = up_ramp
        y[-(ramp + padding):-padding] = down_ramp
        y[0:padding] = 0
        y[-padding:] = 0
    else:
        y[0:ramp] = up_ramp
        y[-ramp:] = down_ramp

    return(y)


def plot_win_1d(y):
    """Helper function for visualising 1d windows

    """

    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, y.size, num=y.size)
    line, = plt.plot(x, y, linewidth=2)

    plt.show()
