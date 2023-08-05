#!/usr/bin/env python3
#
# Tests the datkit time vector checking methods.
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
import unittest

import numpy as np

import datkit as d


class CheckTimesTest(unittest.TestCase):
    """ Tests methods from the hidden _check_times module. """

    def test_is_increasing(self):

        x = np.linspace(0, 1, 101)
        self.assertTrue(d.is_increasing(x))
        y = np.linspace(10, 20, 1001)
        self.assertTrue(d.is_increasing(y))
        z = np.concatenate(([-8, -6, -4], np.arange(5), np.arange(10, 30)))
        self.assertTrue(d.is_increasing(z))

        x[50] = 2
        self.assertFalse(d.is_increasing(x))
        x[50] = x[49]
        self.assertFalse(d.is_increasing(x))
        x[50] = x[49] + 1e-6
        self.assertTrue(d.is_increasing(x))
        y[500] = -1
        self.assertFalse(d.is_increasing(y))
        z = np.concatenate((np.arange(5), [-8, -6, -4], np.arange(10, 30)))
        self.assertFalse(d.is_increasing(z))

        self.assertRaisesRegex(ValueError, 'two values', d.is_increasing, [])
        self.assertRaisesRegex(ValueError, 'two values', d.is_increasing, [0])
        x = np.array([[0, 1], [1, 2]])
        self.assertRaisesRegex(ValueError, '1-d', d.is_increasing, x)
        x = [(0, 1), [1, 2]]
        self.assertRaisesRegex(ValueError, '1-d', d.is_increasing, x)

    def test_is_regularly_increasing(self):

        x = np.linspace(0, 1, 101)
        self.assertTrue(d.is_regularly_increasing(x))
        y = np.linspace(10, 20, 1001)
        self.assertTrue(d.is_regularly_increasing(y))
        z = np.concatenate(([-8, -6, -4], np.arange(5), np.arange(10, 30)))
        self.assertFalse(d.is_regularly_increasing(z))

        self.assertTrue(d.is_regularly_increasing(-x[::-1]))
        self.assertFalse(d.is_regularly_increasing(x[::-1]))
        self.assertFalse(d.is_regularly_increasing(-x))
        x = np.linspace(-1, 1, 101)
        self.assertTrue(d.is_regularly_increasing(x))
        self.assertFalse(d.is_regularly_increasing(np.concatenate((x, [2]))))
        self.assertFalse(d.is_regularly_increasing(np.concatenate(([-2], x))))
        self.assertFalse(d.is_regularly_increasing(np.logspace(1, 10, 11)))

        self.assertTrue(d.is_regularly_increasing(-13 + np.arange(10) / 100))

    def test_sampling_interval(self):

        self.assertEqual(d.sampling_interval(np.arange(10)), 1)
        self.assertEqual(d.sampling_interval(np.arange(10) / 10), 0.1)
        self.assertEqual(d.sampling_interval(np.arange(10) / 100), 0.01)
        self.assertAlmostEqual(
            d.sampling_interval(-13 + np.arange(10) / 100), 0.01)
        self.assertAlmostEqual(
            d.sampling_interval(-13 + np.arange(1000) / 100), 0.01)
        x = -13 + np.arange(1000) / 100
        print(len(x))

        self.assertRaisesRegex(ValueError, 'two', d.sampling_interval, [])
        self.assertRaisesRegex(ValueError, 'two', d.sampling_interval, [0])
        self.assertEqual(d.sampling_interval((-5, 5)), 10)
        x = np.array([[0, 1], [1, 2]])
        self.assertRaisesRegex(ValueError, '1-d', d.sampling_interval, x)
        x = [(0, 1), [1, 2]]
        self.assertRaisesRegex(ValueError, '1-d', d.sampling_interval, x)


if __name__ == '__main__':
    unittest.main()
