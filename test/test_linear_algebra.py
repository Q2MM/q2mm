from __future__ import print_function

import os
import sys

import unittest
import numpy as np

src_dir = os.path.abspath("q2mm")
sys.path.append(src_dir)
import linear_algebra


class MakeHessian(object):
    def __init__(self):
        self.out = []

    def write(self, str):
        self.out.append(str)

    def __str__(self):
        return "".join(self.out)


example_sq3 = np.asarray([[-2, -4, 2], [-4, 1, 2], [2, 2, 5]])
# Eigenvalues are sorted in ascending order to match the output from np.linalg.eigh
# which is used by linear_algebra methods (decompose in particular)
evals_sq3 = np.asarray([-5.51082, 3.659278, 5.851542])
# Matrix rows must be ordered such that eigenvectors correspond to correct eigenvalues
evec_sq3 = np.asarray(
    [[-3.06513, -2.19028, 1], [2.82137, -3.49173, 1], [0.0770902, 0.348681, 1]]
)
evec_normd_sq3 = np.array([evec / np.linalg.norm(evec) for evec in evec_sq3])


class TestLinearAlgebra(unittest.TestCase):
    def test_replace_neg_eigenvalue(self):
        repl_evals = linear_algebra.replace_neg_eigenvalue(evals_sq3)
        np.testing.assert_allclose([1.0, evals_sq3[1], evals_sq3[2]], repl_evals)
        multi_neg_evals = [
            -5,
            -2.45,
            -1,
            -0.2,
            0.3,
            0.459,
            0.5,
            1.8,
            2.3,
            4.6,
            5,
            9.87456,
        ]
        multi_neg_evals_repl = [
            1.0,
            0.0,
            0.0,
            0.0,
            0.3,
            0.459,
            0.5,
            1.8,
            2.3,
            4.6,
            5,
            9.87456,
        ]
        np.testing.assert_allclose(
            multi_neg_evals_repl,
            linear_algebra.replace_neg_eigenvalue(multi_neg_evals, zer_out_neg=True),
            err_msg="Replaced eigenvalues do not match. Failed to replace excess negative values with zero.",
        )

    def test_reform_hessian(self):
        evals, evecs = linear_algebra.decompose(example_sq3)
        reformed_sq3 = linear_algebra.reform_hessian(evals, evecs)
        np.testing.assert_allclose(example_sq3, reformed_sq3, err_msg="Hessian is not reformed properly.")
