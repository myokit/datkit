#
# Methods to find values at points.
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
import numpy as np

import datkit


def abs_max_on(times, values, t0, t1, include_left=True, include_right=False):
    """
    Returns a tuple ``(t_max, v_max)`` corresponding to the maximum value in
    ``values`` on the interval from ``t0`` to ``t1``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    i = i + np.argmax(np.abs(values[i:j]))
    return times[i], values[i]


def index(times, t, ttol=1e-9):
    """
    Returns the index of time ``t`` in ``times``.

    A ``ValueError`` will be raised if time ``t`` cannot be found in ``times``.
    Two times will be regarded as equal if they are within ``ttol`` of each
    other.
    """
    # Check t is within range
    if t < times[0]:
        if abs(times[0] - t) <= ttol:
            return 0
        raise ValueError(
            f'Time t is outside the provided range: {t} < {times[0]}.')
    if t > times[-1]:
        if abs(t - times[-1]) <= ttol:
            return len(times) - 1
        raise ValueError(
            f'Time t is outside the provided range: {t} > {times[-1]}.')

    # Find index and return
    i = np.searchsorted(times, t)   # times[i - 1] < t <= times[i]
    i = i if i == 0 or times[i] - t < t - times[i - 1] else i - 1
    if abs(times[i] - t) > ttol:
        raise ValueError(f'Time t={t} is not present in the data. Nearest'
                         f' is {times[i]} at index {i}.')
    return i


def index_near(times, t):
    """
    Returns the index of time ``t`` in ``times``, or the index of the nearest
    value to it.

    If ``t`` is outside the range of ``times`` by more than half a sampling
    interval (as returned by :meth:`datkit.sampling_interval`), a
    ``ValueError`` will be raised.
    """
    # Check t is within range
    if t < times[0]:
        dt = datkit.sampling_interval(times)
        if 2 * (times[0] - t) < dt:
            return 0
        raise ValueError(
            f'Time t is too far outside the provided range: {t} < {times[0]}')
    elif t > times[-1]:
        dt = datkit.sampling_interval(times)
        if 2 * (t - times[-1]) < dt:
            return len(times) - 1
        raise ValueError(
            f'Time t is too far outside the provided range: {t} > {times[-1]}')

    # Find index and return
    i = np.searchsorted(times, t)   # times[i - 1] < t <= times[i]
    return i if i == 0 or times[i] - t < t - times[i - 1] else i - 1


def index_on(times, t0, t1, include_left=True, include_right=False):
    """
    Returns a tuple ``(i0, i1)`` corresponding to the interval from ``t0`` to
    ``t1`` in ``times``.

    By default, the interval is taken as ``t0 <= times < t1``, but this can be
    customized using ``include_left`` and ``include_right``.

    If any of the points are outside of range, the interval returned will be
    smaller or even empty.
    """
    if len(times) < 1:
        raise ValueError('Times must contain at least one value.')
    if t1 < t0:
        raise ValueError('Time t1 must be greater than or equal to t0.')
    i = np.searchsorted(times, t0)
    j = np.searchsorted(times, t1)
    if (not include_left) and i < len(times) and times[i] == t0:
        i += 1
    if include_right and j < len(times) and times[j] == t1:
        j += 1
    return i, j


def max_on(times, values, t0, t1, include_left=True, include_right=False):
    """
    Returns a tuple ``(t_max, v_max)`` corresponding to the maximum value in
    ``values`` on the interval from ``t0`` to ``t1``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    i = i + np.argmax(values[i:j])
    return times[i], values[i]


def mean_on(times, values, t0, t1, include_left=True, include_right=False):
    """
    Returns the mean of ``values`` on the interval from ``t0`` to ``t1``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    return np.mean(values[i:j])


def min_on(times, values, t0, t1, include_left=True, include_right=False):
    """
    Returns a tuple ``(t_min, v_min)`` corresponding to the minimum value in
    ``values`` on the interval from ``t0`` to ``t1``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    i = i + np.argmin(values[i:j])
    return times[i], values[i]


def value_at(times, values, t, ttol=1e-9):
    """
    Returns the value at the given time point.

    A ``ValueError`` will be raised if time ``t`` cannot be found in ``times``.
    Two times will be regarded as equal if they are within ``ttol`` of each
    other.
    """
    return values[index(times, t, ttol=ttol)]


def value_near(times, values, t):
    """
    Returns the value nearest the given time point, if present in the data.

    A ``ValueError`` will be raised if no time near ``t`` can be found in
    ``times`` (see :meth:`index_near`).
    """
    return values[index_near(times, t)]

