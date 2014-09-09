# miscellaneous functions
# -*- coding: utf-8 -*-

import numpy as _np
import os as _os
import psyutils as _pu


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


def draw_box(size, channel='r', width=4):
    """Make a box of a given size that can be placed into images to highlight
    a  region of interest. The middle of the box is transparent (i.e. alpha 0)
    to show what's in the region of interest.

    Args:
        size (tuple or scalar):
            the size of the box in pixels; either square if a scalar is passed
            or (w, h) from tuple.
        channel (string):
            specify box colour according to colour channel ('r', 'g', 'b')
        width (int):
            width of box lines in pixels.

    Returns:
        a numpy array with shape [size, size, 4].
    """

    if channel == 'r':
        chan = 0
    elif channel == 'g':
        chan = 1
    elif channel == 'b':
        chan = 2
    else:
        raise ValueError("don't know what colour channel to use")

    w, h = _pu.image.parse_size(size)
    box = _np.zeros((h, w, 4))
    box[0:h, 0:width, chan] = 1.
    box[0:h, -width:, chan] = 1.
    box[0:width, 0:w, chan] = 1.
    box[-width:, 0:w, chan] = 1.

    box[0:h, 0:width, 3] = 1.
    box[0:h, -width:, 3] = 1.
    box[0:width, 0:w, 3] = 1.
    box[-width:, 0:w, 3] = 1.

    return(box)


def pix_per_deg(viewing_distance, screen_wh_px, screen_wh_cm,
                average_wh=True):
    """Return the number of pixels per degree of visual angle for a given
    viewing distance of a screen of some resolution and size.

    Note: this assumes a constant viewing distance, so there will be an error
    that increases with eccentricity. For example, at a viewing distance of
    60 cm, something 30 degrees eccentric will be at a distance of 69 cm
    (60 / np.cos(30 * np.pi / 180)), if presented on a flat screen. At that
    viewing distance, the number of pixels per degree will be higher (46
    compared to 40 for the example monitor below) --- i.e. about a 13
    percent size error at 30 degrees.

    Args:
        viewing_distance (float):
            the viewing distance of the screen (screen to subject's eye) in cm.
        screen_wh_px (tuple):
            the width and height of the screen in pixels.
        screen_wh_cm (tuple):
            the width and height of the screen in cm.
        average_wh (boolean, default True):
            if true, computes pix per deg based on the average of the
            width and height.
            If false, returns a tuple (width, height).

    Returns:
        float: the number of pixels per degree of visual angle, assuming a
        constant distance.
        or if average_wh=False, a 2 element numpy array.

    Example::
        dist = 60
        px = (1920, 1080)
        cm = (52, 29)
        pu.misc.pix_per_deg(60, (1920, 1080), (52, 29))
        # gives 40.36 pixels per degree.
    """

    wh_px = _np.array(screen_wh_px)
    wh_cm = _np.array(screen_wh_cm)

    ppd = _np.pi * (wh_px) / _np.arctan(wh_cm / viewing_distance / 2.) / 360.

    if average_wh is True:
        res = ppd.mean()
    elif average_wh is False:
        res = ppd

    return(res)


def create_project_folder(project_name, path=None, gitignore=True):
    """ Create a new project folder in the current working directory containing
    all subfolders.

    Args:
        project_name (string):
            the name for the project.
        path (string, optional):
            an optional path name for the directory containing the project.
            If not provided, project folder will be created in the current
            working directory.
        gitignore (boolean, defaults True):
            if True, create a .gitignore file with common options of files to
            ignore for the git version control system.

    Example:
        Make a directory for the project "awesome-science" in /home/usr/::
        create_project_folder('awesome-science', path='/home/usr/')

    """

    if path is None:
        root_dir = _os.getcwd()
    else:
        root_dir = path

    # check if project directory exists, create it (taken from
    # https://stackoverflow.com/questions/273192/
    # check-if-a-directory-exists-and-create-it-if-necessary)
    def ensure_dir(d):
        if not _os.path.exists(d):
            _os.makedirs(d)

    top_dir = _os.path.join(root_dir, project_name)
    ensure_dir(top_dir)

    # code subdirectories:
    ensure_dir(_os.path.join(top_dir, 'code', 'analysis'))
    ensure_dir(_os.path.join(top_dir, 'code', 'experiment'))
    ensure_dir(_os.path.join(top_dir, 'code', 'stimuli'))
    ensure_dir(_os.path.join(top_dir, 'code', 'unit-tests'))

    # other subdirectories:
    ensure_dir(_os.path.join(top_dir, 'documentation'))
    ensure_dir(_os.path.join(top_dir, 'figures'))
    ensure_dir(_os.path.join(top_dir, 'notebooks'))
    ensure_dir(_os.path.join(top_dir, 'publications'))
    ensure_dir(_os.path.join(top_dir, 'raw-data'))
    ensure_dir(_os.path.join(top_dir, 'results'))

    if gitignore is True:
        create_gitignore(top_dir)


def create_gitignore(path=None):
    """Creates a `.gitignore` file with common options in the current working
    directory, or in `path`.

    Args:
    path (string, optional):
        an optional path name for the directory to place the file.
        If not provided, .gitignore will be created in the current
        working directory.

    Example:
        Make a .gitignore in /home/usr/::
        create_gitignore(path='/home/usr/')

    """

    if path is None:
        root_dir = _os.getcwd()
    else:
        root_dir = path

    fname = _os.path.join(root_dir, '.gitignore')

    f = open(fname, mode='x')

    f.write('.gitignore \n'
            '.Rhistory \n'
            '.Rproj* \n'
            '*.Rproj \n'
            '.DS_Store* \n'
            '*.odt \n'
            '*.aux \n'
            '*.log \n'
            '*.out \n'
            '*.gz \n'
            '*.sublime-project \n'
            '*.sublime-workspace \n'
            '*.ipynb \n'
            '*.pyc \n'
            '*.pdf \n'
            '*.png \n'
            '*.svg \n'
            '*.jpg \n'
            '*.jpeg \n'
            '*.doc \n'
            '*.docx')
    f.close()
    # close(f)
