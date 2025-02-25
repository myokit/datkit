#!/usr/bin/env python3
#
# Tests the datkit points methods.
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
import unittest

import numpy as np

import datkit as d


class PointsTest(unittest.TestCase):
    """ Tests methods from the hidden _points module. """

    def test_abs_max_on(self):
        t = np.linspace(0, 2, 101)
        v = np.cos(t * np.pi)
        self.assertEqual(d.abs_max_on(t, v, 0, 1), (0, 1))
        self.assertEqual(d.abs_max_on(t, v, 0.5, 1), (t[49], v[49]))
        self.assertEqual(d.abs_max_on(t, v, 0.5, 1, True, True), (1, -1))
        self.assertEqual(d.abs_max_on(t, v, 0.6, 1.5), (1, -1))
        self.assertEqual(d.abs_max_on(t, v, 1.5, 2), (t[99], v[99]))
        self.assertEqual(d.abs_max_on(t, v, 1.5, 2, False, True), (2, 1))

    def test_data_on(self):
        t = [0, 1, 2, 3, 4, 5, 6, 7]
        v = [10, 11, 12, 13, 14, 15, 16, 17]
        self.assertEqual(d.data_on(t, v, 3, 5), ([3, 4], [13, 14]))
        self.assertEqual(d.data_on(t, v, 4), ([4, 5, 6, 7], [14, 15, 16, 17]))
        self.assertEqual(d.data_on(t, v, t1=2), ([0, 1], [10, 11]))
        self.assertEqual(d.data_on(t, v, t1=2, include_right=True),
                         ([0, 1, 2], [10, 11, 12]))

    def test_iabs_max_on(self):
        t = np.linspace(0, 2, 101)
        v = np.cos(t * np.pi)
        self.assertEqual(d.iabs_max_on(t, v, 0, 1), 0)
        self.assertEqual(d.iabs_max_on(t, v, 0.5, 1), 49)
        self.assertEqual(d.iabs_max_on(t, v, 0.5, 1, True, True), 50)
        self.assertEqual(d.iabs_max_on(t, v, 0.6, 1.5), 50)
        self.assertEqual(d.iabs_max_on(t, v, 1.5, 2), 99)
        self.assertEqual(d.iabs_max_on(t, v, 1.5, 2, False, True), 100)

    def test_imax_on(self):
        t = np.linspace(0, 2, 101)
        v = np.cos(t * np.pi)
        self.assertEqual(d.imax_on(t, v, 0, 1), 0)
        self.assertEqual(d.imax_on(t, v, 0.5, 1), 25)
        self.assertEqual(d.imax_on(t, v, 0.6, 1.5), 74)
        self.assertEqual(d.imax_on(t, v, 1.5, 2), 99)
        self.assertEqual(d.imax_on(t, v, 1.5, 2, False, True), 100)

    def test_imin_on(self):
        t = np.linspace(0, 2, 101)
        v = np.cos(t * np.pi)
        self.assertEqual(d.imin_on(t, v, 0, 1), 49)
        self.assertEqual(d.imin_on(t, v, 0.5, 2), 50)
        self.assertEqual(d.imin_on(t, v, 0.5, 1.5), 50)
        self.assertEqual(d.imin_on(t, v, 1.5, 2), 75)
        self.assertEqual(d.imin_on(t, v, 1.5, 2, False), 76)

    def test_index(self):

        # Simple tests
        times = np.arange(0, 10)
        self.assertEqual(d.index(times, 0), 0)
        self.assertEqual(d.index(times, 4), 4)
        self.assertEqual(d.index(times, 9), 9)
        with self.assertRaisesRegex(ValueError, 'not present'):
            d.index(times, 0.5)
        times = np.linspace(0, 1, 101)
        self.assertEqual(d.index(times, 0), 0)
        self.assertEqual(d.index(times, 1), 100)
        self.assertEqual(times[d.index(times, 1)], 1)
        self.assertEqual(times[d.index(times, 0.01)], 0.01)
        self.assertEqual(times[d.index(times, 0.51)], 0.51)
        with self.assertRaisesRegex(ValueError, '-0.01 < 0.0'):
            d.index(times, -0.01)
        with self.assertRaisesRegex(ValueError, '2 > 1.0'):
            d.index(times, 2)

        # Test finding within a tolerance
        self.assertEqual(d.index(times, 1e-10), 0)
        self.assertEqual(d.index(times, 0.01 + 1e-10), 1)
        self.assertEqual(d.index(times, 0.01 - 1e-10), 1)
        self.assertEqual(d.index(times, 0.03 + 1e-9), 3)
        self.assertEqual(d.index(times, 0.03 - 1e-9), 3)
        with self.assertRaisesRegex(ValueError, 'not present'):
            d.index(times, 0.01 + 2e-9)
        with self.assertRaisesRegex(ValueError, 'not present'):
            d.index(times, 0.01 - 2e-9)
        self.assertEqual(d.index(times, 0.02 + 1e-7, ttol=1e-7), 2)
        self.assertEqual(d.index(times, 0.02 - 1e-7, ttol=1e-7), 2)
        with self.assertRaisesRegex(ValueError, 'not present'):
            d.index(times, 0.02 + 1e-7, ttol=1e-8)
        with self.assertRaisesRegex(ValueError, 'not present'):
            d.index(times, 0.02 - 1e-7, ttol=1e-8)

        # Edges
        times = np.arange(0, 10, 2)
        self.assertEqual(d.index(times, 0), 0)
        self.assertEqual(d.index(times, -1e-10), 0)
        self.assertEqual(d.index(times, 1e-10), 0)
        self.assertEqual(d.index(times, -1e-9), 0)
        self.assertRaisesRegex(ValueError, 'range', d.index, times, -2e-9)
        self.assertEqual(d.index(times, 8), 4)
        self.assertEqual(d.index(times, 8 - 1e-10), 4)
        self.assertEqual(d.index(times, 8 + 1e-10), 4)
        self.assertEqual(d.index(times, 8 + 9e-10), 4)
        self.assertRaisesRegex(ValueError, 'range', d.index, times, 8 + 2e-9)
        times = 0.1 * (-25 + np.arange(0, 100, 2))
        self.assertEqual(d.index(times, -2.5), 0)
        self.assertEqual(d.index(times, -2.5 - 1e-10), 0)
        self.assertEqual(d.index(times, -2.5 + 1e-10), 0)
        self.assertEqual(d.index(times, -2.5 - 9e-10), 0)
        self.assertRaisesRegex(ValueError, 'range',
                               d.index, times, -2.5 - 2e-9)
        self.assertEqual(d.index(times, 7.3), 49)
        self.assertEqual(d.index(times, 7.3 - 1e-10), 49)
        self.assertEqual(d.index(times, 7.3 + 1e-10), 49)
        self.assertEqual(d.index(times, 7.3 + 9e-10), 49)
        self.assertRaisesRegex(ValueError, 'range', d.index, times, 7.3 + 2e-9)

        # Any sequence is accepted
        self.assertEqual(d.index(tuple(times), 7.3), 49)

    def test_index_crossing(self):

        # Simple test
        values = [4, 5, 6, 7, 8, 6, 7, 8, 9]
        self.assertEqual(d.index_crossing(values, 6.5), (2, 3))
        self.assertEqual(d.index_crossing(values, 8.5), (7, 8))
        self.assertRaisesRegex(
            ValueError, 'No crossing', d.index_crossing, values, 1)
        self.assertRaisesRegex(
            ValueError, 'No crossing', d.index_crossing, values, 4)

        # Quadratic and cubic
        values = np.linspace(-5, 5, 100)**2
        self.assertRaisesRegex(
            ValueError, 'No crossing', d.index_crossing, values)
        values = (np.linspace(-5, 5, 100) - 3)**3
        self.assertEqual(d.index_crossing(values), (79, 80))
        self.assertTrue(values[79] < 0, values[80] > 0)
        values = -(np.linspace(-5, 5, 100) - 2)**3
        self.assertEqual(d.index_crossing(values), (69, 70))
        self.assertTrue(values[69] > 0, values[70] < 0)

        # Annoying case 1: starting or ending at value
        self.assertRaisesRegex(
            ValueError, 'No crossing', d.index_crossing, [4, 5, 6], 4)
        self.assertRaisesRegex(
            ValueError, 'No crossing', d.index_crossing, [4, 4, 4, 5, 6], 4)
        self.assertRaisesRegex(
            ValueError, 'No crossing', d.index_crossing, [4, 5, 6], 6)
        self.assertRaisesRegex(
            ValueError, 'No crossing', d.index_crossing, [4, 5, 6, 6, 6], 6)
        values = [3, 3, 3, 4, 5, 4, 3, 2, 1, 2, 3, 3, 3]
        self.assertEqual(d.index_crossing(values, 3), (5, 7))

        # Annoying case 2: being flat at the selected value
        values = [9, 9, 8, 7, 6, 5, 5, 5, 5, 4, 3, 2, 2]
        self.assertEqual(d.index_crossing(values, 5), (4, 9))

    def test_index_near(self):

        # Exact matches
        times = np.arange(0, 10)
        self.assertEqual(d.index_near(times, 0), 0)
        self.assertEqual(d.index_near(times, 4), 4)
        self.assertEqual(d.index_near(times, 9), 9)

        # Near matches
        self.assertEqual(d.index_near(times, 0.1), 0)
        self.assertEqual(d.index_near(times, 2.1), 2)
        self.assertEqual(d.index_near(times, 3.5), 3)

        # Outside of range
        times = np.arange(0, 20) / 2
        self.assertEqual(d.index_near(times, -0.1), 0)
        self.assertEqual(d.index_near(times, -0.24999), 0)
        self.assertRaisesRegex(
            ValueError, 'range', d.index_near, times, -0.251)
        self.assertEqual(d.index_near(times, 9.6), 19)
        self.assertEqual(d.index_near(times, 9.7499), 19)
        self.assertRaisesRegex(ValueError, 'range', d.index_near, times, 9.751)

        # Any sequence is accepted
        self.assertEqual(d.index_near(tuple(times), 9.6), 19)
        self.assertEqual(d.index_near(list(times), 9.6), 19)

    def test_index_on(self):
        t = np.arange(0, 10)
        self.assertEqual(d.index_on(t, 2, 4), (2, 4))
        self.assertEqual(d.index_on(t, 2, 4.1), (2, 5))
        self.assertEqual(d.index_on(t, 0.1, 5), (1, 5))
        self.assertEqual(d.index_on(t, -5, 4), (0, 4))

        self.assertEqual(d.index_on(t, 2, 4, include_left=False), (3, 4))
        self.assertEqual(d.index_on(t, -5, 4, include_left=False), (0, 4))
        self.assertEqual(d.index_on(t, 0, 4, include_left=False), (1, 4))
        self.assertEqual(d.index_on(t, 2, 4, include_right=True), (2, 5))
        self.assertEqual(d.index_on(t, 2, 3.9, include_right=True), (2, 4))
        self.assertEqual(d.index_on(t, 2, 100), (2, 10))
        self.assertEqual(d.index_on(t, 2, 100, include_right=True), (2, 10))

        self.assertEqual(d.index_on(t, 3, 8, True, True), (3, 9))
        self.assertEqual(d.index_on(t, 3, 8, False, True), (4, 9))
        self.assertEqual(d.index_on(t, 3, 8, True, False), (3, 8))
        self.assertEqual(d.index_on(t, 3, 8, False, False), (4, 8))

        self.assertEqual(d.index_on(t, -3, 88, True, True), (0, 10))
        self.assertEqual(d.index_on(t, -3, 88, False, True), (0, 10))
        self.assertEqual(d.index_on(t, -3, 88, True, False), (0, 10))
        self.assertEqual(d.index_on(t, -3, 88, False, False), (0, 10))

        self.assertEqual(d.index_on(t, -9, -3, True, True), (0, 0))
        self.assertEqual(d.index_on(t, -9, -3, False, True), (0, 0))
        self.assertEqual(d.index_on(t, -9, -3, True, False), (0, 0))
        self.assertEqual(d.index_on(t, -9, -3, False, False), (0, 0))

        self.assertEqual(d.index_on(t, 12, 18, True, True), (10, 10))
        self.assertEqual(d.index_on(t, 12, 18, False, True), (10, 10))
        self.assertEqual(d.index_on(t, 12, 18, True, False), (10, 10))
        self.assertEqual(d.index_on(t, 12, 18, False, False), (10, 10))

        self.assertEqual(d.index_on(t, 0, 0), (0, 0))
        self.assertEqual(d.index_on(t, 4, 4), (4, 4))
        self.assertEqual(d.index_on(t, -4, -4), (0, 0))
        self.assertEqual(d.index_on(t, 10, 10), (10, 10))
        self.assertEqual(d.index_on(t, 12, 12), (10, 10))

        self.assertRaisesRegex(
            ValueError, 'at least one', d.index_on, [], 2, 4)
        self.assertEqual(d.index_on([3], 2, 4), (0, 1))
        self.assertEqual(d.index_on([3], 2, 3), (0, 0))
        self.assertRaisesRegex(ValueError, 'greater than', d.index_on, t, 3, 2)

        t = np.arange(4, 40, 2)
        self.assertEqual(d.index_on(t, 8, 16), (2, 6))
        t = np.arange(-6, 18, 3)
        self.assertEqual(d.index_on(t, -3, 9), (1, 5))

        # Values not specified
        t = np.arange(0, 20, 2)
        self.assertEqual(d.index_on(t), (0, 10))
        self.assertEqual(d.index_on(t, include_left=False), (0, 10))
        self.assertEqual(d.index_on(t, include_right=True), (0, 10))
        self.assertEqual(
            d.index_on(t, include_left=False, include_right=True), (0, 10))
        self.assertEqual(d.index_on(t, 3), (2, 10))
        self.assertEqual(d.index_on(t, None, 10), (0, 5))

        # Any sequence is accepted
        self.assertEqual(d.index_on(tuple(t), 3), (2, 10))
        self.assertEqual(d.index_on(list(t), 3), (2, 10))

    def test_max_on(self):
        t = np.linspace(0, 2, 101)
        v = np.cos(t * np.pi)
        self.assertEqual(d.max_on(t, v, 0, 1), (0, 1))
        self.assertEqual(d.max_on(t, v, 0.5, 1), (t[25], v[25]))
        self.assertEqual(d.max_on(t, v, 0.6, 1.5), (t[74], v[74]))
        self.assertEqual(d.max_on(t, v, 1.5, 2), (t[99], v[99]))
        self.assertEqual(d.max_on(t, v, 1.5, 2, False, True), (t[100], v[100]))

    def test_mean_on(self):
        t = np.arange(1, 11)
        self.assertEqual(d.mean_on(t, t, 1, 11), 5.5)
        self.assertEqual(d.mean_on(t, t, 4, 8), 5.5)
        self.assertEqual(d.mean_on(t, t, 4, 8, False), 6)
        self.assertEqual(d.mean_on(t, t, 4, 8, True, True), 6)
        v = -3 + 8 * t[::-1]
        self.assertEqual(d.mean_on(t, v, 1, 11), 41)
        self.assertEqual(d.mean_on(t, v, 4, 8), 41)
        self.assertEqual(d.mean_on(t, v, 4, 8, False), 37)
        self.assertEqual(d.mean_on(t, v, 4, 8, True, True), 37)

    def test_min_on(self):
        t = np.linspace(0, 2, 101)
        v = np.cos(t * np.pi)
        self.assertEqual(d.min_on(t, v, 0, 1), (t[49], v[49]))
        self.assertEqual(d.min_on(t, v, 0.5, 2), (t[50], v[50]))
        self.assertEqual(d.min_on(t, v, 0.5, 1.5), (t[50], v[50]))
        self.assertEqual(d.min_on(t, v, 1.5, 2), (t[75], v[75]))
        self.assertEqual(d.min_on(t, v, 1.5, 2, False), (t[76], v[76]))

    def test_time_crossing(self):
        t = np.linspace(1, 5, 100)
        v = np.sin(t) + 1
        self.assertLess(abs(d.time_crossing(t, v, 1) - np.pi), 1e-7)
        self.assertRaises(ValueError, d.time_crossing, t, v)
        t = np.linspace(0, 5, 100)
        self.assertRaises(ValueError, d.time_crossing, t, np.cos(t) - 1)
        t, v = [2, 3, 4, 5], [10, 20, 30, 40]
        self.assertEqual(d.time_crossing(t, v, 25), 3.5)
        self.assertEqual(d.time_crossing(t, v, 31), 4.1)
        t, v = [4, 5, 6, 7], [50, 40, 30, 20]
        self.assertEqual(d.time_crossing(t, v, 25), 6.5)
        self.assertEqual(d.time_crossing(t, v, 31), 5.9)

    def test_value_at(self):
        t = np.arange(0, 10)
        self.assertEqual(d.value_at(t, t, 0), 0)
        self.assertEqual(d.value_at(t, t, 5), 5)
        self.assertEqual(d.value_at(t, t, 9), 9)
        v = 20 + 2 * t
        self.assertEqual(d.value_at(t, v, 0), 20)
        self.assertEqual(d.value_at(t, v, 5), 30)

    def test_value_interpolated(self):
        t, v = [2, 3, 4, 5, 6, 7], [5, 0, 3, -1, 4, 8]
        self.assertEqual(d.value_interpolated(t, v, 2), 5)
        self.assertEqual(d.value_interpolated(t, v, 4), 3)
        self.assertEqual(d.value_interpolated(t, v, 7), 8)
        self.assertEqual(d.value_interpolated(t, v, 4.5), 1)
        self.assertEqual(d.value_interpolated(t, v, 5.5), 1.5)
        self.assertAlmostEqual(d.value_interpolated(t, v, 2.2), 4)
        self.assertAlmostEqual(d.value_interpolated(t, v, 6.9), 7.6)
        self.assertRaisesRegex(ValueError, 'entries in times',
                               d.value_interpolated, t, v, 1.9)
        self.assertRaisesRegex(ValueError, 'entries in times',
                               d.value_interpolated, t, v, 7.1)
        t, v = [0, 1, 2], [6, 6, 6]
        self.assertEqual(d.value_interpolated(t, v, 0), 6)
        self.assertEqual(d.value_interpolated(t, v, 1), 6)
        self.assertEqual(d.value_interpolated(t, v, 2), 6)

    def test_value_near(self):
        t = np.arange(0, 10)
        self.assertEqual(d.value_near(t, t, 0), 0)
        self.assertEqual(d.value_near(t, t, 5), 5)
        self.assertEqual(d.value_near(t, t, 9), 9)
        self.assertEqual(d.value_near(t, t, 0.1), 0)
        self.assertEqual(d.value_near(t, t, 5.7), 6)
        self.assertEqual(d.value_near(t, t, 8.9), 9)
        v = 20 + 2 * t
        self.assertEqual(d.value_at(t, v, 0), 20)
        self.assertEqual(d.value_at(t, v, 5), 30)


if __name__ == '__main__':
    unittest.main()
