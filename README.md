#Psy Utils

The ``psyutils`` package is a collection of utility functions useful
for generating visual stimuli and analysing the results of
psychophysical experiments. It will eventually include modules
for data munging, image processing (such as filtering images
in the fourier domain, for example for creating
filtered noise) and other stimulus-related functions.

It is **NOT** an experimental display software package. It is intended
to compliment something like [PsychoPy](http://www.psychopy.org).

## Subpackages

 * ``data``. The ``data`` subpackage includes functions for data munging,
  for example for formatting data usefully for import in ``pandas``
  or to R.
 * ``im_data``. The ``im_data`` subpackage includes images and data for
  testing purposes.
 * ``image``. The ``image`` subpackage includes functions for filtering
  images and creating filtered noise stimuli, and otherwise interacting
  with images.
 * ``misc``. This is a miscellaneous subpackage.

## Example Use

You could do things like this::

    from psyutils import image

    blah blah

## Dependencies

Psyutils depends on numpy, scipy, and scikit-image.

## Testing

You can run unit tests by typing `nosetests -v` (or `nosetests-3.x` for
testing under Python 3) from the command line in the project's parent directory.

## TODO

  * unit tests for filters
  * unit tests for windowing
  * documentation for filters
  * ipython notebook showing cool filtering demos
  * consider moving rad_dist and meshgrid calculations into a common function.
    Must return multiple values (xx, yy, rad_dist).
  * fix cos2d window: I'm using a fairly hacky method that produces strange shapes depending on settings. e.g. try pu.image.cos_win_2d(im_x=128, ramp=20, padding=15).
  * include a fixation cross image (see Gegenfurtner paper).

## Contributions

Thomas Wallis wrote these functions based somewhat on shared code used in Matlab by Peter Bex's lab at the Schepens Eye Research Institute. I also
borrowed some stuff from the scikit-image package. Function abstraction, pythonic niceness and other sweet programming improvements are provided by David Janssen.

## Thanks To

Many of these functions are based on Matlab functions originally written
by Peter Bex. Tom Wallis also thanks Matthias Kümmerer for Python help
and suggestions.
