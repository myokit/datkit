#!/usr/bin/env python3
#
# Tests the datkit spectral analysis methods.
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
import numpy as np

import datkit as d
import datkit.tests


class SpectralTest(datkit.tests.TestCase):
    """ Tests methods from the hidden _spectral module. """

    def test_amplitude_spectrum(self):

        t = np.linspace(0, 10, 1000)
        v = 3 * np.sin(t * (2 * np.pi * 3)) + 10 * np.cos(t * (2 * np.pi * 7))
        f, a = d.amplitude_spectrum(t, v)
        self.assertAlmostEqual(f[30], 2.997)
        self.assertAlmostEqual(f[70], 6.993)
        self.assertLess(a[29], 0.1)
        self.assertAlmostEqual(a[30], 3, 0)
        self.assertLess(a[31], 0.1)
        self.assertLess(a[69], 1)
        self.assertAlmostEqual(a[70], 10, 0)
        self.assertLess(a[71], 1)

        t = np.linspace(0, 10, 1235)
        v = 6 * np.sin(t * (2 * np.pi * 2)) + 4 * np.cos(t * (2 * np.pi * 5))
        f, a = d.amplitude_spectrum(t, v)
        self.assertAlmostEqual(f[20], 1.998, 3)
        self.assertAlmostEqual(f[50], 4.996, 3)
        self.assertLess(a[19], 0.11)
        self.assertAlmostEqual(a[20], 6, 0)
        self.assertLess(a[21], 0.11)
        self.assertLess(a[49], 0.2)
        self.assertAlmostEqual(a[50], 4, 0)
        self.assertLess(a[51], 0.2)

        # Test if input is unchanged
        t = np.linspace(0, 10, 123)
        v = 6 * np.sin(t * (2 * np.pi * 2))
        self.assertUnchanged(d.amplitude_spectrum, t, v)

    def test_power_spectral_density(self):

        t = np.linspace(0, 10, 1000)
        v = 3 * np.sin(t * (2 * np.pi)) + 7 * np.cos(t * (2 * np.pi * 3))
        f, psd = d.power_spectral_density(t, v)
        self.assertAlmostEqual(f[10], 0.999, 3)
        self.assertAlmostEqual(f[30], 2.997, 3)

        self.assertLess(psd[9], 0.01)
        self.assertAlmostEqual(psd[10], 45, 1)
        self.assertLess(psd[11], 0.01)
        self.assertLess(psd[29], 0.3)
        self.assertAlmostEqual(psd[30], 245, 0)
        self.assertLess(psd[31], 0.3)

        # Test if input is unchanged
        t = np.linspace(0, 10, 123)
        v = 6 * np.sin(t * (2 * np.pi * 2))
        self.assertUnchanged(d.power_spectral_density, t, v)


if __name__ == '__main__':
    import unittest
    unittest.main()
