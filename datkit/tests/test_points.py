#!/usr/bin/env python3
#
# Tests the datkit points methods.
#
# This file is part of Datkit.
# See http://myokit.org for copyright, sharing, and licensing details.
#
import unittest

import numpy as np

import datkit as d


class PointsTest(unittest.TestCase):
    """ Tests methods from the hidden _points module. """

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


if __name__ == '__main__':
    unittest.main()
