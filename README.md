#Psy Utils

The ``psyutils`` package is a collection of utility functions useful
for generating visual stimuli and analysing the results of
psychophysical experiments. It will eventually include modules
for data munging, image processing (such as filtering images
in the fourier domain, for example for creating
filtered noise) and other stimulus-related functions.

It is **NOT** an experimental display software package. It is intended
to compliment something like `PsychoPy <http://www.psychopy.org>`.

## Subpackages

 * ``data``. The ``data`` subpackage includes functions for data munging,
  for example for formatting data usefully for import in ``pandas``
  or to R.
 * ``im_data``. The ``im_data`` subpackage includes images and data for
  testing purposes.
 * ``image``. The ``image`` subpackage includes functions for filtering
  images and creating filtered noise stimuli, and otherwise interacting
  with images.
 * ``misc``. This is a miscellaneous subpackage for things like ramp
  functions, e.g. creating 1D Gaussian or cosine windows for
  ramping stimuli on and off.

## Example Use

You could do things like this::

    from psyutils import image

    blah blah

## Dependencies

Psyutils depends on numpy, scipy, and scikit-image.

## Testing

You can run unit tests by typing `nosetests -v` from the command line
in the project's parent directory.

## Contributions

Thomas Wallis wrote these functions based on shared code used in Matlab
by Peter Bex's lab at the Schepens Eye Research Institute. I also
borrowed some stuff from the scikit-image package.

## Thanks To

Many of these functions are based on Matlab functions originally written
by Peter Bex. Tom Wallis also thanks Matthias Kümmerer for Python help
and suggestions.
