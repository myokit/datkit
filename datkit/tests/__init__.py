#!/usr/bin/env python3
#
# Test module
#
# This file is part of Datkit.
# For copyright, sharing, and licensing, see https://github.com/myokit/datkit/
#
import os
import unittest

# The test directory
DIR_TEST = os.path.abspath(os.path.dirname(__file__))


class TestCase(unittest.TestCase):
    """
    Adds methods to ``unittest.TestCase``.
    """

    def array_args_unchanged(self, func, *args):
        """
        Test if sequence-type arguments are unchanged by the given function.
        """
        import numpy as np

        new_args = []
        seq_args = {}
        for k, arg in enumerate(args):
            new_args.append(arg)
            if not isinstance(arg, (str, bytes, dict)):
                try:
                    len(arg)
                except TypeError:
                    pass
                else:
                    new_args[k] = np.array(arg)
                    seq_args[k] = np.array(arg, copy=True)
        if len(seq_args) == 0:
            return True

        # Call, check if unchanged
        func(*new_args)
        for k, b in seq_args.items():
            a = new_args[k]
            if a.shape != b.shape:
                return False
            if np.any(a != b):
                return False
        return True

    def assertUnchanged(self, func, *args):
        """
        Fail if the given function changes the sequence-type arguments.
        """
        self.assertTrue(self.array_args_unchanged(func, *args))


del os, unittest
