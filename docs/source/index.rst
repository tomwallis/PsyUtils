.. Psyutils documentation master file, created by
   sphinx-quickstart on Thu Aug 28 14:55:04 2014.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Psyutils's documentation!
====================================

The ``psyutils`` package is a collection of utility functions useful
for generating visual stimuli and analysing the results of psychophysical experiments. It is a work in progress, and changes as I work. It includes various helper functions for dealing with images, including creating and applying various filters in the frequency domain.

It is **NOT** an experimental display software package. It is intended
to compliment something like [PsychoPy](http://www.psychopy.org).

It is intended only for internal use in Tom's science at this stage. It is provided publicly to facilitate reproducibility of research. *Use these functions at your own risk.* The unit-testing is incomplete.

Subpackages
-------------

 * ``image``. The ``image`` subpackage includes functions for filtering
  images and creating filtered noise stimuli, and otherwise interacting
  with images.

 * ``dist``: functions for creating probability distributions over axes. Currently just used to create filters.

 * ``im_data``. The ``im_data`` subpackage includes images and data for
  testing purposes.

 * ``misc``. This is a miscellaneous subpackage.


.. toctree::
   :maxdepth: 2

.. automodule:: image
    :members:

.. automodule:: misc
    :members:

.. automodule:: im_data
    :members:

.. automodule:: dist
    :members:

Example Use
-----------

You could do things like this::

    import numpy as np
    import psyutils as pu
    im = np.random.uniform(size=(256, 256))
    filt = pu.image.make_filter_lowpass(im_size, cutoff=8)
    im2 = pu.image.filter_image(im, filt)
    pu.image.show_im(im2)

Dependencies
------------

Psyutils depends on numpy, scipy, and scikit-image.

Testing
-------
You can run unit tests by typing `nosetests -v` (or `nosetests-3.x` for
testing under Python 3) from the command line in the project's parent directory.


Contributions
-------------

Thomas Wallis wrote these functions based somewhat on shared code used in
Matlab by Peter Bex's lab at the Schepens Eye Research Institute. I also
borrowed some stuff from the scikit-image package. Function abstraction,
pythonic niceness and other sweet programming improvements were suggested
by David Janssen.

Thanks To
---------

Many of these functions are based on Matlab functions originally written
by Peter Bex. Tom Wallis also thanks Matthias Kümmerer for Python help
and suggestions.



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

