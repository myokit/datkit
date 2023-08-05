#
# SetupTools script for Datkit
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
from setuptools import setup, find_packages


# Get version number
import os
import sys
sys.path.append(os.path.abspath('datkit'))
from _datkit_version import __version__ as version  # noqa
sys.path.pop()
del os, sys


# Load text for description and license
with open('README.md') as f:
    readme = f.read()


# Go!
setup(
    # See https://python-packaging.readthedocs.io/en/latest/index.html
    # for details of what goes in here.

    # Module name (lowercase)
    name='datkit',

    # Version
    version=version,

    # Description
    description='A bunch of scripts',
    long_description=readme,
    long_description_content_type='text/markdown',

    # Author and license
    license='BSD 3-clause license',
    author='Michael Clerx',
    author_email='michael@myokit.org',

    # URLs
    #url='http://myokit.org',
    project_urls={
        'Bug Tracker': 'https://github.com/myokit/datkit/issues',
        'Documentation': 'http://datkit.readthedocs.io/',
        'Source Code': 'https://github.com/myokit/datkit',
    },

    # Classifiers for pypi
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering',
    ],

    # Packages to include
    packages=find_packages(include=('datkit', 'datkit.*')),

    # Include non-python files (via MANIFEST.in)
    #include_package_data=True,

    # Python version
    python_requires='>=3.7',

    # List of dependencies
    install_requires=[
        'numpy',
    ],

    # Optional extras
    extras_require={
        'docs': [
            'sphinx>=1.7.4',        # Doc generation
        ],
        'dev': [
            'coverage',             # Coverage checking
            'flake8>=3',            # Style checking
        ],
    },

    # Unit tests
    test_suite='datkit.tests',
)
