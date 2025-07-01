#
# Methods to inspect time series in the frequency domain.
# Note that much more advanced methods are available e.g. in scipy.
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
import numpy as np

import datkit


def amplitude_spectrum(times, values):
    """
    Calculates the amplitude spectrum of a regularly spaced time series
    ``(times, current)``, returning a tuple ``(frequency, amplitude)``.

    Example::

        t = np.linspace(0, 10, 1000)
        v = 3 * np.sin(t * (2 * np.pi * 2)) + 10 * np.cos(t * (2 * np.pi * 3))
        f, a = amplitude_spectrum(t, v)

        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(t, v)
        ax = fig.add_subplot(2, 1, 2)
        ax.set_xlim(0, 10)
        ax.stem(f, a)
        plt.show()

    """
    dt = datkit.sampling_interval(times)
    n = len(times)
    # Select only the positive frequencies
    m = n // 2
    # FFT, normalised by number of point
    a = np.absolute(np.fft.fft(values)[:m]) / n
    # Only using one half, so multiply values of mirrored points by 2
    a[1:-1] *= 2
    # Get corresponding frequencies and return
    f = np.fft.fftfreq(n, dt)[:m]
    return f, a


def power_spectral_density(times, values):
    """
    Estimates the power spectral density of a regularly spaced time series
    ``(times, current)``, returning a tuple ``(frequency, density)``.

    For times in units "T" and values in units "V", the returned frequencies
    will be in units "F=1/T" and the densities in "V^2/F".

    For a less noisy version, use ``scipy.signal.welch(values,
    1 / datkit.sampling_interval(times))`` instead.

    Example::

        t = np.linspace(0, 10, 1000)
        v = 5 * np.sin(t * (2 * np.pi * 2)) + 10 * np.cos(t * (2 * np.pi * 3))
        f, psd = power_spectral_density(t, v)

        fig = plt.figure()
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(t, v)
        ax = fig.add_subplot(2, 1, 2)
        ax.set_yscale('log')
        ax.set_xlim(0, 10)
        ax.plot(f, psd)
        plt.show()

    """
    dt = datkit.sampling_interval(times)
    n = len(times)
    m = n // 2
    p = np.absolute(np.fft.fft(values)[:m])**2 * dt / n
    p[1:-1] *= 2
    f = np.fft.fftfreq(n, dt)[:m]
    return f, p

