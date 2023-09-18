from __future__ import print_function
import copy
import logging
import logging.config
import os
import unittest
import parmed
import numpy as np

import constants as co
import datatypes


class MakeInput(object) :

    def __init__(self) :
        self.out = []

    def write(self, str) :
        self.out.append(str)

    def __str__(self) :
        return "".join(self.out)


class TestLinearAlgebra(unittest.TestCase):

    def test_loadstruct(self) :

        struct = parmed.load_file(gro_filename, skip_bonds=True)
        self.assertEqual(len(struct.atoms),41)