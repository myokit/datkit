#!/usr/bin/env python3
#
# Tests testing methods.
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
import numpy as np

import datkit.tests


def mixed(a, b, c, d, e):
    return True


def change3(a, b, c, d, e):
    c[3] += 1


class ArgsUnchangedTest(datkit.tests.TestCase):
    """ Tests methods from ``datkit.tests``. """

    def test_args_unchanged(self):
        self.assertTrue(self.array_args_unchanged(
            mixed, 1, 2, 3, 4, np.array([1, 2, 3])))
        self.assertUnchanged(mixed, 1, 4, np.array([1, 2, 3]), 2, 'hi')
        self.assertFalse(self.array_args_unchanged(
            change3, 1, 2, np.array([1, 2, 3, 4]), 4, 5))
        self.assertFalse(self.array_args_unchanged(
            change3, 1, 2, [1, 2, 3, 4], 4, 5))
        self.assertFalse(self.array_args_unchanged(
            change3, 'hello', 2, (1, 2, 3, 4), 4, 5))


if __name__ == '__main__':
    import unittest
    unittest.main()
