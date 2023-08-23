"""
This module contains a function to read Gaussian09 output file
"""
from __future__ import division, print_function, absolute_import

import numpy as np

__all__ = [ 'parse_fchk' ]

def parse_fchk(filename):
    """
    Parse Gaussian09 formated checkpoint file for coordinates
    and Hessian

    Parameters
    ----------
    filename : string
        the name of input file

    Returns
    -------
    numpy.ndarray
        the optimized xyz coordinates in the checkpoint file
    numpy.ndarray
        the Hessian matrix in the checkpoint file
    """
    def _parse_array(f, startline, endline):
        arr = []
        line = "None"
        while line and not line.startswith(startline) :
            line = f.readline()
        while line and not line.startswith(endline):
            line = f.readline()
            if not line.startswith(endline):
                arr.extend(line.strip().split())
        return np.array(arr, dtype=float)

    crds = None
    hess = None
    with open(filename, "r") as f :
        # First the coordinates
        crds = _parse_array(f, 'Current cartesian coordinates', 'Force Field')
        # Then the Hessian
        hess = _parse_array(f, 'Cartesian Force Constants', 'Dipole Moment')

    # Make the Hessian in square form
    i = 0
    n = len(crds)
    hess_sqr = np.zeros([n, n])
    for j in range(n):
        for k in range(j+1):
            hess_sqr[j,k] = hess[i]
            hess_sqr[k,j] = hess[i]
            i += 1

    natoms = int(n / 3)
    return crds.reshape([natoms,3]), hess_sqr
