#
# Methods to check and analyze lists of times.
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
import numpy as np


def is_increasing(times):
    """
    Checks if each value in ``times`` is greater than (not equal to) the last.
    """
    times = np.asarray(times)
    if len(times.shape) != 1:
        raise ValueError('Times must be a 1-d numpy array.')
    if len(times) < 2:
        raise ValueError('Times must contain at least two values.')

    return np.all(times[1:] > times[:-1])


def is_regularly_increasing(times, reltol=1e-12):
    """
    Checks if the difference between successive times is always the same
    strictly positive value.

    Checks that ``dt``, the value returned by :meth:`sampling_interval`, is
    strictly positive and that
    ``abs((times[i + 1] - times[i]) / dt - 1) < reltol`` for all ``i``.
    """
    times = np.asarray(times)
    dt = sampling_interval(times)
    if dt <= 0:
        return False
    d = np.abs(np.diff(times) / dt - 1)
    return np.all(d < reltol)


def sampling_interval(times):
    """
    Returns the sampling interval, assuming that ``times`` is regularly spaced.
    """
    times = np.asarray(times)
    if len(times.shape) != 1:
        raise ValueError('Times must be a 1-d numpy array.')
    if len(times) < 2:
        raise ValueError('Times must contain at least two values.')

    # Note: This is slower, but gets rid of numerical noise
    if len(times) < 100:
        return np.mean(np.diff(times))
    return np.mean(np.diff(times[:100]))

