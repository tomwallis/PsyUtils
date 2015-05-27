from distutils.core import setup

setup(
    name='PsyUtils',
    version='1.0.0',
    author='Thomas S. A. Wallis',
    author_email='thomas.wallis@uni-tuebingen.de',
    packages=['psyutils', 'psyutils.image'],
    #scripts=['bin/stowe-towels.py','bin/wash-towels.py'],
    #url='http://pypi.python.org/pypi/TowelStuff/',
    license='LICENSE.txt',
    description='Utility functions for psychophysical experiments and \
                stimuli.',
    long_description=open('README.md').read(),
    install_requires=[
        # 'Python >= 2.7.0',
        'numpy >= 1.8.0',
        'scipy >= 0.13.0',
        'matplotlib',
        'scikit-image >= 0.9.0'
    ],
)
