#
# Methods to smooth a time series.
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
import numpy as np

import datkit as d


def gaussian_smoothing(times, values, w=None, t=None):
    """
    Applies a Gaussian smoothing filter to ``v``, using a window of either
    ``w`` samples or a number derived from the duration ``t``.

    If ``w`` is given, it must be a strictly positive and odd integer,
    indicating the number of data points to include in the window. If a
    duration ``t`` is given, it will be converted to an integer width using the
    method :meth:`window_size`.

    The Gaussian kernel is determined as ``np.exp(-i**2)``, where ``i`` ranges
    from -2 to +2 in the window.

    Returns a new time series ``x, y`` of length ``len(times) - w + 1``.
    """
    times, values = np.asarray(times), np.asarray(values)
    if len(times) != len(values):
        raise ValueError('Times and values vectors must have same size.')

    w = window_size(times, w, t)
    k = np.exp(-np.linspace(-2, 2, w)**2)
    t = times[w // 2: -(w // 2)]
    return t, np.convolve(values, k, 'valid') / sum(k)


def haar_downsample(times, values, repeats=1):
    """
    Returns a downsampled signal created by successive averaging of adjacent
    samples, similar to a Haar wavelet.

    In each pass:

    - If the signal length is odd, the final sample is omitted
    - The first two samples are averaged, then the next two, etc.

    Returns a new and downsampled time series ``(times_2, values_2)`` of length
    ``len(times) // 2**repeats``.
    """
    times, values = np.asarray(times), np.asarray(values)
    if len(times) != len(values):
        raise ValueError('Times and values vectors must have same size.')

    for r in range(repeats):
        if len(times) % 2:
            times, values = times[:-1], values[:-1]
        times = 0.5 * (times[::2] + times[1::2])
        values = 0.5 * (values[::2] + values[1::2])
    return times, values


def moving_average(times, values, w=None, t=None):
    """
    Applies a moving average filter to ``v``, using a window of either ``w``
    samples or a number derived from the duration ``t``.

    If ``w`` is given, it must be a strictly positive and odd integer,
    indicating the number of data points to include in the window. If a
    duration ``t`` is given, it will be converted to an integer width using the
    method :meth:`window_size`.

    Returns a new time series ``x, y`` of length ``len(times) - w + 1``.
    """
    times, values = np.asarray(times), np.asarray(values)
    if len(times) != len(values):
        raise ValueError('Times and values vectors must have same size.')

    w = window_size(times, w, t)
    t = times[w // 2: -(w // 2)]
    return t, np.convolve(values, np.ones(w), 'valid') / w


def window_size(times, w=None, t=None):
    """
    Returns a window size of either ``w`` samples or duration ``t``.

    If ``w`` is given, it must be a strictly positive odd integer.

    If ``t`` is given, the used width will be the nearest odd integer to
    ``t / dt``, where ``dt`` is the sampling interval. The nearest odd integer
    is defined as ``1 + 2 * ((t / dt) // 2)``.
    """
    if w is None:
        if t is None:
            raise ValueError(
                'No window size specified: w and t are both None.')

        t = float(t)
        dt = d.sampling_interval(times)
        if t / dt < 2.9:
            raise ValueError(
                'Invalid window size: must be at least 3 times the sampling'
                f' interval {dt}.')

        w = int(1 + 2 * ((t / dt) // 2))

    elif t is None:
        w_org = w
        w = int(round(w))
        if w < 3 or w % 2 == 0:
            raise ValueError(
                'Invalid window size: must be an odd number and at least'
                f' 3 or greater. Got {w_org}.')

    else:
        raise ValueError(
            'Two window sizes specified: w and t are both not None.')

    if w > len(times):
        raise ValueError(
            'Invalid window size: Must be no greater than length of times'
            f' vector ({len(times)}), got {w}.')
    return w

