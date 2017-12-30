from distutils.core import setup

setup(
    name='PsyUtils',
    version='1.3.2',
    author='Thomas S. A. Wallis',
    author_email='thomas.wallis@uni-tuebingen.de',
    packages=['psyutils'],
    url='http://github.com/tomwallis/PsyUtils',
    license='LICENSE.txt',
    description='Utility functions for psychophysical experiments and \
                stimuli.',
    long_description=open('README.md').read(),
    install_requires=[
        'numpy >= 1.8.0',
        'scipy >= 0.13.0',
        'matplotlib',
        'pillow >= 4.0.0',
        'scikit-image >= 0.12.0',
        'seaborn >= 0.7.0',
        'pycircstat',
        'nose'
    ],
)
