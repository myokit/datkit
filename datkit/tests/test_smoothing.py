#!/usr/bin/env python3
#
# Tests the datkit smoothing methods.
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
import unittest

import numpy as np

import datkit as d


# Extra output
debug = False


class SmoothingTest(unittest.TestCase):
    """ Tests methods from the hidden _smoothing module. """

    def test_gaussian_smoothing(self):

        # Numerical tests with ones: should return all ones
        n, w = 10, 3
        t = np.arange(n)
        v = np.ones(n)
        x, y = d.gaussian_smoothing(t, v, w)
        self.assertEqual(len(x), len(t) - w + 1)
        self.assertEqual(len(x), len(y))
        self.assertTrue(np.all(y == 1))

        n, w = 11, 7
        t = np.arange(n)
        v = np.ones(n)
        x, y = d.gaussian_smoothing(t, v, w)
        self.assertEqual(len(x), len(t) - w + 1)
        self.assertEqual(len(x), len(y))
        self.assertTrue(np.allclose(y, 1))

        n, w = 5, 5
        t = np.arange(n)
        v = np.ones(n)
        x, y = d.gaussian_smoothing(t, v, w)
        self.assertEqual(len(x), len(t) - w + 1)
        self.assertEqual(len(x), len(y))
        self.assertTrue(np.allclose(y, 1))

        # Numerical test with increasing sequences
        t = np.arange(5)
        x, y = d.gaussian_smoothing(t, t, 3)
        self.assertEqual(list(x), list(t[1:-1]))
        # Note: Because the kernel is symetric, and because the series is
        # regularly increasing, this returns the middle value!
        self.assertTrue(np.allclose(y, [1, 2, 3]))

        t = np.arange(2, 17, 3)
        x, y = d.gaussian_smoothing(t, t, 3)
        self.assertEqual(list(x), list(t[1:-1]))
        self.assertTrue(np.allclose(y, [5, 8, 11]))

        # Numerical test with randomish sequence
        t = np.arange(5)
        v = np.array([3, 5, 2, 1, 7])
        x, y = d.gaussian_smoothing(t, v, 3)
        self.assertEqual(list(x), list(t[1:-1]))
        self.assertEqual(len(x), len(y), 3)
        k = np.exp(-np.linspace(-2, 2, 3)**2)
        k /= sum(k)
        self.assertAlmostEqual(y[0], np.sum(k * v[0:3]))
        self.assertAlmostEqual(y[1], np.sum(k * v[1:4]))
        self.assertAlmostEqual(y[2], np.sum(k * v[2:5]))

        # Real application: denoising
        r = np.random.default_rng(1)
        t = np.linspace(0, 1, 1001)
        v = np.sin(t * 2 * np.pi)
        v += 0.02 * np.cos(t * 150 * np.pi)
        v += r.normal(0, 0.02, v.shape)
        x, y = d.gaussian_smoothing(t, v, w=21)
        z = np.sin(x * 2 * np.pi)
        self.assertLess(np.max(np.abs(y - z)), 0.03)

        if debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(t, v)
            plt.plot(x, y)
            plt.plot(x, z)
            plt.show()

    def test_haar_downsample(self):

        # Numerical tests with ones: should return all ones
        t, v = np.arange(10), np.ones(10)
        x, y = d.haar_downsample(t, v)
        self.assertEqual(len(x), len(y), 5)
        self.assertTrue(np.all(y == 1))
        x, y = d.haar_downsample(x, y)
        self.assertEqual(len(x), len(y), 2)
        self.assertTrue(np.all(y == 1))
        x, y = d.haar_downsample(x, y)
        self.assertEqual(len(x), len(y), 1)
        self.assertTrue(np.all(y == 1))
        x, y = d.haar_downsample(x, y)
        self.assertEqual(len(x), len(y), 0)
        x, y = d.haar_downsample(x, y)
        self.assertEqual(len(x), len(y), 0)

        t, v = np.arange(5), np.ones(5)
        x, y = d.haar_downsample(t, v, 2)
        self.assertEqual(len(x), len(y), 1)
        self.assertTrue(np.all(y == 1))
        x, y = d.haar_downsample(t, v, 3)
        self.assertEqual(len(x), len(y), 0)
        x, y = d.haar_downsample(t, v, 4)
        self.assertEqual(len(x), len(y), 0)

        # Numerical tests with proper values
        t = np.arange(10)
        v = [5, 1, 8, 12, 3, 3, 1, 9, 2, 7]
        x, y = d.haar_downsample(t, v)
        self.assertEqual(list(x), [0.5, 2.5, 4.5, 6.5, 8.5])
        self.assertEqual(list(y), [3, 10, 3, 5, 4.5])
        x, y = d.haar_downsample(t, v, 2)
        self.assertEqual(list(x), [1.5, 5.5])
        self.assertEqual(list(y), [6.5, 4])
        x, y = d.haar_downsample(t, v, 3)
        self.assertEqual(list(x), [3.5])
        self.assertEqual(list(y), [5.25])

        # Real application: denoising
        r = np.random.default_rng(1)
        t = np.linspace(0, 1, 1001)
        v = np.sin(t * 2 * np.pi)
        v += 0.02 * np.cos(t * 150 * np.pi)
        v += r.normal(0, 0.02, v.shape)
        x, y = d.haar_downsample(t, v, 3)
        z = np.sin(x * 2 * np.pi)
        self.assertLess(np.max(np.abs(z - y)), 0.04)

        if debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(t, v)
            plt.plot(x, y)
            plt.plot(x, z)
            plt.show()

    def test_moving_average(self):

        # Numerical tests with ones: should return all ones
        n, w = 10, 3
        t = np.arange(n)
        v = np.ones(n)
        x, y = d.moving_average(t, v, w)
        self.assertEqual(len(x), len(t) - w + 1)
        self.assertEqual(len(x), len(y))
        self.assertTrue(np.all(y == 1))

        n, w = 11, 7
        t = np.arange(n)
        v = np.ones(n)
        x, y = d.moving_average(t, v, w)
        self.assertEqual(len(x), len(t) - w + 1)
        self.assertEqual(len(x), len(y))
        self.assertTrue(np.all(y == 1))

        n, w = 5, 5
        t = np.arange(n)
        v = np.ones(n)
        x, y = d.moving_average(t, v, w)
        self.assertEqual(len(x), len(t) - w + 1)
        self.assertEqual(len(x), len(y))
        self.assertTrue(np.all(y == 1))

        # Numerical test with increasing sequences
        t = np.arange(5)
        x, y = d.moving_average(t, t, 3)
        self.assertEqual(list(x), list(t[1:-1]))
        self.assertEqual(list(y), [1, 2, 3])

        t = np.arange(2, 17, 3)
        x, y = d.moving_average(t, t, 3)
        self.assertEqual(list(x), list(t[1:-1]))
        self.assertEqual(list(y), [5, 8, 11])

        # Real application: denoising
        r = np.random.default_rng(1)
        t = np.linspace(0, 1, 1001)
        v = np.sin(t * 2 * np.pi)
        v += 0.02 * np.cos(t * 150 * np.pi)
        v += r.normal(0, 0.02, v.shape)
        x, y = d.moving_average(t, v, w=21)
        z = np.sin(x * 2 * np.pi)
        self.assertLess(np.max(np.abs(y - z)), 0.02)

        if debug:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(t, v)
            plt.plot(x, y)
            plt.plot(x, z)
            plt.show()

    def test_window_size(self):

        self.assertTrue(False)


if __name__ == '__main__':
    print('Add -v for more debug output')
    import sys
    if '-v' in sys.argv:
        debug = True
    unittest.main()
