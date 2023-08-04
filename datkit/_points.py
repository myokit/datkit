#
# Methods to find values at points.
#
# This file is part of Datkit.
# See http://myokit.org for copyright, sharing, and licensing details.
#
import numpy as np


def index(times, t, lpad=None, rpad=None, ttol=1e-9):
    """
    Returns the index of time ``t`` in ``times``.

    If ``lpad`` and/or ``rpad`` are given, a tuple ``(lo, hi)`` will be
    returned where ``lo`` is the index of ``t - lpad`` and ``hi`` is the index
    of ``t + rpad``. Both padding values must be ``None``, zero, or positive.
    Note that this equates to a half-open interval: the range ``lo:hi`` will
    not include ``t + rpad`` (or ``t``, if ``rpad`` is ``None``).

    An error will be raised if time ``t`` cannot be found in ``times`` (or
    similarly if ``t - lpad`` or ``t + rpad`` cannot be found. Times will be
    regarded as equal if they are within ``ttol`` of each other.
    """
    # Check t is within range
    if t < times[0] and abs(times[0] - t) > ttol:
        raise ValueError('Time t is outside the provided range:'
                         f' {t} < {times[0]}.')
    if t > times[-1] and abs(t - times[-1]) > ttol:
        raise ValueError('Time t is outside the provided range:'
                         f' {t} > {times[-1]}.')

    # Handle single-value version
    if lpad is None and rpad is None:
        i = np.searchsorted(times, t)   # times[i - 1] < t <= times[i]
        i = i if i == 0 or times[i] - t < t - times[i - 1] else i - 1
        if abs(times[i] - t) > ttol:
            raise ValueError(f'Time t={t} is not present in the data. Nearest'
                             f' is {times[i]} at index {i}.')
        return i

    # Handle padded version
    t0 = t if lpad is None else t - lpad
    t1 = t if rpad is None else t + rpad
    if t0 > t or t1 < t:
        raise ValueError('Left and right padding must be 0 or positive.')
    if t0 < times[0] and abs(times[0] - t0) > ttol:
        raise ValueError('Time minus lpad is outside the provided range:'
                         f' {t0} < {times[0]}.')
    if t1 > times[-1] and abs(t1 - times[-1]) > ttol:
        raise ValueError('Time plus rpad is outside the provided range:'
                         f' {t1} > {times[-1]}.')

    i0 = np.searchsorted(times, t0)
    i0 = i0 if i0 == 0 or times[i0] - t0 < t0 - times[i0 - 1] else i0 - 1
    if abs(times[i0] - t0) > ttol:
        raise ValueError(f'Left-padded time {t0} not present in the data.'
                         f' Nearest is {times[i0]} at index {i0}.')
    i1 = np.searchsorted(times, t1)   # times[i - 1] < t <= times[i]
    i1 = i1 if i1 == 0 or times[i1] - t1 < t1 - times[i1 - 1] else i1 - 1
    if abs(times[i1] - t1) > ttol:
        raise ValueError(f'Right-padded time {t1} not present in the data.'
                         f' Nearest is {times[i1]} at index {i1}.')
    return i0, i1


def value_at(times, values, t, lpad=None, rpad=None, ttol=1e-9):
    """
    Returns the value at the given time point, if present in the data.

    If left and right padding is provided, the mean of the indicated segment
    is returned.
    """
    if lpad is None and rpad is None:
        return values[index_of(times, t, ttol=ttol)]

    i, j = index_of(times, t, lpad, rpad, ttol)
    return np.mean(values[i:j])
