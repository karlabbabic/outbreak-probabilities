#
from setuptools import setup, find_packages


def get_version():
    """
    Load version string from version.py without importing the package.
    """
    version_ns = {}
    with open('src/outbreak_probabilities/version_info.py') as f:
        exec(f.read(), {}, version_ns)
    return version_ns['__version__']

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
    version= get_version(),

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
