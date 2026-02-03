[![Unit tests](https://github.com/myokit/datkit/actions/workflows/unit-tests-ubuntu.yml/badge.svg)](https://github.com/myokit/datkit/actions/workflows/unit-tests-ubuntu.yml)
[![Documentation](https://readthedocs.org/projects/datkit/badge/?version=latest)](https://datkit.readthedocs.io/?badge=latest)

# Datkit

This module contains some simple methods that are useful when analysing time 
series data.

Building only on numpy, they reliably let you do things like:

- Find the indices strictly corresponding to a point, or the start and end
  time of a range (including end points, half-open, etc)
- Find the index closest to a point _not_ appearing in the data, and
  interpolate the value at that point.
- Perform some action (max, min, mean, absolute max etc) on a range
- Check if a series is regularly spaced, or strictly non-decreasing, and
  obtain the sampling time
- Smooth using sliding windows, gaussian blur, and others
- Get an amplitude or power spectrum of periodic data

In short, things you can easily do with numpy, but without having to remember
the syntax, and with unit tests for that particular operation.

The code is tested on a recent version of Ubuntu & Python 3, but is so simple
that it should work everywhere else too.

## Installation

To install the latest release from PyPI, use

```
pip install datkit
```

## Installation for development

To install from the repo, use e.g.
```
python setup.py install -e .
```

Tests can then be run with
```
python -m unittest
```

And docs can be built with
```
cd docs
make clean html
```
