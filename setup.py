from distutils.core import setup

setup(
    name='PsyUtils',
    version='0.1.0',
    author='Thomas S. A. Wallis',
    author_email='thomas.wallis@uni-tuebingen.de',
    packages=['psyutils', 'psyutils.data', 'psyutils.image', 'psyutils.misc', 'psyutils.im_data'],
    #scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    #url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Utility functions for psychophysical experiments and stimuli.',
    long_description=open('README.txt').read(),
    install_requires=[
        'Python >= 2.7.0',
        'numpy >= 1.8.0',
        'scipy >= 0.13.0',
        'matplotlib',
        'skimage >= 0.9.0'
    ],
)
