# miscellaneous functions
# -*- coding: utf-8 -*-

import numpy as _np


def fixation_cross():
    """Return a 256 square numpy array containing a rendering of the
    fixation cross recommended in Thaler et al for low dispersion and
    microsaccade rate. You could rescale this to the appropriate size (outer
    ring should be 0.6 dva in diameter and inner ring 0.2 dva).

    Example:
        Our stimulus display has 40 pixels per degree of visual angle::
        from skimage import transform
        sz = round(40 * 0.6)
        fixation_cross = transform.resize(pu.misc.fixation_cross(), (sz,sz))

    Reference:
        Thaler, L., Schütz, A. C., Goodale, M. A., & Gegenfurtner, K. R. (2013)
        What is the best fixation target? The effect of target shape on
        stability of fixational eye movements. Vision Research, 76(C), 31–42.
    """

    outer_rad = 128
    inner_rad = int((0.2 / 0.6)*outer_rad)  # inner is 0.2

    def _draw_oval(radius):
        im = _np.ones((radius*2, radius*2))
        x = _np.linspace(-radius, radius, num=radius*2)
        xx, yy = _np.meshgrid(x, x)
        rad_dist = (xx**2 + yy**2)**0.5
        im[rad_dist <= radius] = 0
        return(im)

    im = _draw_oval(outer_rad)
    im[outer_rad - inner_rad:outer_rad + inner_rad, :] = 1
    im[:, outer_rad - inner_rad:outer_rad + inner_rad] = 1
    im[outer_rad-inner_rad:outer_rad+inner_rad,
       outer_rad-inner_rad:outer_rad+inner_rad] = _draw_oval(inner_rad)
    return(im)
