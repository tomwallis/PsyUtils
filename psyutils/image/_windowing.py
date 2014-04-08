# Functions for windowing, including images.


def cos_win_1d(length,
               cent='default',
               ramp='default',
               padding=0):
    """Create a vector of length ``length`` containing a 1D cosine window
    where the centre of the window is set to 1 and the ramps go down to
    zero (with optional zero padding). This can be useful for e.g. adjusting
    the alpha channel of a stimulus over frames so that the stimulus is
    smoothly ramped on and off.

    This function will do some basic checking to see that your parameters
    make sense but it is not thorough. If parameters are provided as integers
    they will be converted to floats.

    Example:
        Create a vector of length 200 where the value is 1 for the central
        100 samples, on and off ramps of length 45 and 5 samples of zero
        padding at either end::
        $ window = cos_win_1d(length = 200, cent = 100, ramp = 45, padding = 5)
        import matplotlib.pyplot as plt
        x = np.linspace(0, y.size, num = v.size)
        line, = plt.plot(x, y, linewidth=2)
        plt.show()

    Args:
        length (int or float): the length of the window function to return.
        cent (int or float): the number of samples at 1, centred in length.
            Defaults to length * 0.8.
        ramp (int or float): the size of each ramp.
            Defaults to (length-cent) / 2.
        padding (int or float): the size of zero padding at the edges.
            Defaults to zero.

    Returns:
        window (float): a vector containing the windowing kernel,
        normalised 0--1.

    """

    import numpy as np
    from numpy import allclose

    length = float(length)
    mid = length / 2.0

    if cent is 'default':
        cent = length * 0.8
    else:
        cent = float(cent)
    if ramp is 'default':
        ramp = (length - cent) / 2.0
    else:
        ramp = float(ramp)

    padding = int(padding)

    # do a check for params making sense:
    tot = cent + (ramp * 2.0) + (padding * 2.0)

    if allclose(tot, length) is False:
        raise ValueError("Your parameters add up to " + str(tot) + " but you "
                         "asked for length " + str(length))

    v = np.arange(0, length, dtype=np.float64)
    dist_centre = v - mid
    #     cent = length - (2.0 * ramp) - (2.0 * padding)

    v[abs(dist_centre) <= (cent / 2.0)] = 1  # set middle to 1
    #vect(abs(distFromCentre)>=(radius-ramp_length/2)) = 0;

    # set padding range to zero:
    v[0:padding] = 0
    v[-padding:] = 0

    # create the ramps:
    up_ramp = np.sin(np.linspace(0, 1, round(ramp)) * np.pi/2.0)
    down_ramp = np.cos(np.linspace(0, 1, round(ramp)) * np.pi/2.0)

    # place into v:
    v[padding:(padding+ramp)] = up_ramp
    v[-(ramp + padding):-padding] = down_ramp
    return(v)


def plot_win_1d(y):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, y.size, num=y.size)
    line, = plt.plot(x, y, linewidth=2)

    plt.show()
