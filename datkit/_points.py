#
# Methods to find values at points.
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
import numpy as np

import datkit


def abs_max_on(times, values, t0=None, t1=None, include_left=True,
               include_right=False):
    """
    Returns a tuple ``(t_max, v_max)`` corresponding to the maximum value in
    ``abs(values)`` on the interval from ``t0`` to ``t1``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    i = i + np.argmax(np.abs(values[i:j]))
    return times[i], values[i]


def data_on(times, values, t0=None, t1=None, include_left=True,
            include_right=False):
    """
    Returns a tuple ``(times2, values2)`` corresponding to the interval from
    ``t0`` to ``t1`` in ``times``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    return times[i:j], values[i:j]


def iabs_max_on(times, values, t0=None, t1=None, include_left=True,
                include_right=False):
    """
    Returns the index ``i`` corresponding to the maximum of ``abs(values)`` on
    the interval from ``t0`` to ``t1``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    return i + np.argmax(np.abs(values[i:j]))


def imax_on(times, values, t0=None, t1=None, include_left=True,
            include_right=False):
    """
    Returns the index ``i`` corresponding to the maximum value on the interval
    from ``t0`` to ``t1``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    return i + np.argmax(values[i:j])


def imin_on(times, values, t0=None, t1=None, include_left=True,
            include_right=False):
    """
    Returns the index ``i`` corresponding to the maximum value on the interval
    from ``t0`` to ``t1``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    return i + np.argmin(values[i:j])


def index(times, t, ttol=1e-9):
    """
    Returns the index of time ``t`` in ``times``, assuming ``times`` is a
    non-decreasing sequence.

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


def index_crossing(values, value=0):
    """
    Returns the lowest two indices ``i`` and ``j`` for which ``values`` crosses
    the given ``value`` (going either from below to above, or vice versa).

    For example ``datkit.index([0, 1, 2, 3, 4], 2.5)`` returns ``(2, 3)``.

    The method is best applied to smooth (denoised) data.

    A ``ValueError`` is raised if no crossing can be found, or
    """
    # Get sign of values - value, as either -1, 0, or 1
    # This means we can't use numpy's sign function.
    v = np.asarray(values) - value
    s = np.zeros(v.shape)
    s[v > 0] = 1
    s[v < 0] = -1
    # Find first non-zero
    i = np.where(s != 0)[0]
    if len(i) > 0:
        i = i[0]
        # Find first opposing sign
        j = np.where(s == -s[i])[0]
        if len(j) > 0:
            j = j[0]
            # Find last value with original sign
            i = np.where(s[:j] == s[i])[0][-1]
            return i, j
    raise ValueError(f'No crossing of {value} found in array.')


def index_near(times, t):
    """
    Returns the index of time ``t`` in ``times``, or the index of the nearest
    value to it, assuming ``times`` is a non-decreasing sequence.

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


def index_on(times, t0=None, t1=None, include_left=True, include_right=False):
    """
    Returns a tuple ``(i0, i1)`` corresponding to the interval from ``t0`` to
    ``t1`` in ``times``, assuming ``times`` is a non-decreasing sequence.

    By default, the interval is taken as ``t0 <= times < t1``, but this can be
    customized using ``include_left`` and ``include_right``.

    If one or both points are out of range, indices corresponding to the first
    and/or last entries in ``times`` are returned. Note that this may result in
    an empty interval if ``t0 < t1 < times[0]`` or ``times[1] < t0 < t1``.

    If ``t0`` is ``None``, the first index will be ``0``, regardless of the
    value of ``include_left``. If ``t1`` is ``None`` the second index will be
    ``len(times)``, regardless of the value of ``include_right``.
    """
    if len(times) < 1:
        raise ValueError('Times must contain at least one value.')
    if t0 is None:
        t0 = times[0] - 1
    if t1 is None:
        t1 = times[-1] + 1
    if t1 < t0:
        raise ValueError('Time t1 must be greater than or equal to t0.')
    i = np.searchsorted(times, t0)
    j = np.searchsorted(times, t1)
    if (not include_left) and i < len(times) and times[i] == t0:
        i += 1
    if include_right and j < len(times) and times[j] == t1:
        j += 1
    return i, j


def max_on(times, values, t0=None, t1=None, include_left=True,
           include_right=False):
    """
    Returns a tuple ``(t_max, v_max)`` corresponding to the maximum value in
    ``values`` on the interval from ``t0`` to ``t1``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    i = i + np.argmax(values[i:j])
    return times[i], values[i]


def mean_on(times, values, t0=None, t1=None, include_left=True,
            include_right=False):
    """
    Returns the mean of ``values`` on the interval from ``t0`` to ``t1``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    return np.mean(values[i:j])


def min_on(times, values, t0=None, t1=None, include_left=True,
           include_right=False):
    """
    Returns a tuple ``(t_min, v_min)`` corresponding to the minimum value in
    ``values`` on the interval from ``t0`` to ``t1``.

    See also :meth:`index_on`.
    """
    i, j = index_on(times, t0, t1, include_left, include_right)
    i = i + np.argmin(values[i:j])
    return times[i], values[i]


def time_crossing(times, values, value=0):
    """
    Returns the time at which ``values`` first crosses ``value``.

    Specifically, the method linearly interpolates between the entries from
    ``times`` at the indices returned by :meth:`index_crossing`. No assumptions
    are made about ``times`` (other than that it has the same length as
    ``values``), so that arrays representing other quantities can also be
    passed in.

    The method is best applied to smooth (denoised) data.

    See also :meth:`index_crossing`.
    """
    i, j = index_crossing(values, value)
    t0, t1 = times[i], times[j]
    v0, v1 = values[i] - value, values[j] - value
    return t0 - v0 * (t1 - t0) / (v1 - v0)


def value_at(times, values, t, ttol=1e-9):
    """
    Returns ``values[i]`` such that ``times[i]`` is within ``ttol`` of the time
    ``t``.

    A ``ValueError`` will be raised if no such ``i`` can be found.
    """
    return values[index(times, t, ttol=ttol)]


def value_interpolated(times, values, t):
    """
    Returns the value at the given time, obtained by linear interpolation if
    ``t`` is not presesnt in ``times``.

    A ``ValueError`` is raised if no ``i`` can be found such that
    ``times[i] <= t <= times[i + 1]``.
    """
    i = np.searchsorted(times, t)
    n = len(times)
    if n > 0 and i < n and times[i] == t:
        return values[i]
    if i == 0 or i == n:
        raise ValueError(
            'Unable to find entries in times from which to interpolate'
            f' for t={t}.')
    t0, t1 = times[i - 1], times[i]
    v0, v1 = values[i - 1], values[i]
    return v0 + (t - t0) * (v1 - v0) / (t1 - t0)


def value_near(times, values, t):
    """
    Returns ``values[i]`` such that ``times[i]`` is the nearest point to ``t``
    in the data.

    A ``ValueError`` will be raised if no time near ``t`` can be found in
    ``times`` (see :meth:`index_near`).
    """
    return values[index_near(times, t)]

