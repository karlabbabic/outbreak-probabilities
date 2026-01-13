#
from setuptools import setup, find_packages

def get_version():
    """
    Get version number from the abm_model module.

    The easiest way would be to just ``import abm_model``, but note that this may
    fail if the dependencies have not been installed yet. Instead, we've put
    the version number in a simple version_info module, that we'll import here
    by temporarily adding the oxrse directory to the pythonpath using sys.path.
    """
    import os
    import sys

    sys.path.append(os.path.abspath('src.outbreak_probabilities'))
    from version_info import VERSION as version
    sys.path.pop()

    return version

def get_readme():
    """
    Load README.md text for use as description.
    """
    with open('README.md') as f:
        return f.read()

setup(
    # Module name (lowercase)
    name='abm_model',

    # Version
    version=get_version(),

    description='Outbreak Probabilities Project 2025.',

    long_description=get_readme(),

    license='MIT license',

    # author='',

    # author_email='',

    maintainer='Salma Amin, Gemma Marshall, Rayne Alexander, Aniekeme George, Oliver Staples',

    maintainer_email='salma.amin.@dtc.ox.ac.uk', 
     

    url='',

    # Packages to include
    packages=find_packages(include=('src.outbreak_probabilities', 'src.outbreak_probabilities.*')),

    # List of dependencies
    install_requires=[
        # Dependencies go here!
        'numpy',
        'matplotlib',
        'pandas',
        'scipy',
    ],
    extras_require={
        'docs': [
            # Sphinx for doc generation. Version 1.7.3 has a bug:
            'sphinx>=1.5, !=1.7.3',
            # Nice theme for docs
            'sphinx_rtd_theme',
        ],
        'dev': [
            # Flake8 for code style checking
            'flake8>=3',
            'pytest',
            'pytest-cov',
        ],
    },
)
