import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='psyutils',
    version='2.0.0',
    author='Thomas S. A. Wallis',
    author_email='thomas.wallis@uni-tuebingen.de',
    url='http://github.com/tomwallis/PsyUtils',
    license='LICENSE.txt',
    description='Utility functions for psychophysical experiments and \
                stimuli.',
    long_description=long_description,
    long_description_content_type="text/markdown",
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
    packages=setuptools.find_packages()
)
