===========
Psy Utils
===========

The ``psyutils`` package is a collection of utility functions useful
for generating visual stimuli and analysing the results of
psychophysical experiments. It will eventually include modules
for data munging, image processing (such as filtering images
in the fourier domain, for example for creating
filtered noise) and other stimulus-related functions.

It is **NOT** an experimental display software package. It is intended
to compliment something like `PsychoPy <http://www.psychopy.org>`_.

Modules
==========
* ``data``. The ``data`` module includes functions for data munging,
  for example for formatting data usefully for import in ``pandas``
  or to R.
* ``image``. The ``image`` module includes functions for filtering
  images and creating filtered noise stimuli, and otherwise interacting
  with images.
* ``misc``. This is a miscellaneous module for things like ramp
  functions, e.g. creating 1D Gaussian or cosine windows for
  ramping stimuli on and off.

Example Use
============
You could do things like this::

    from psyutils import image

    blah blah


Contributions
============
Thomas Wallis wrote these functions based on shared code used in Matlab
by Peter Bex's lab at the Schepens Eye Research Institute.

Thanks To
============
Many of these functions are based on Matlab functions originally written
by Peter Bex.
