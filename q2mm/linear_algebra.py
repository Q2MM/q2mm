"""_summary_

"""
from __future__ import division, print_function, absolute_import
import argparse
import os
import sys

import numpy as np
import parmed

from datatypes import AmberFF

import logging
import logging.config
import constants as co


# region Generalized

def decompose(matrix:np.ndarray) -> (np.ndarray, np.ndarray):
    """_summary_

    Args:
        matrix (np.ndarray): Matrix to decompose, matrix must be square.

    Returns:
        (np.ndarray, np.ndarray): _description_
    """    
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvalues, eigenvectors

def replace_neg_eigenvalue(eigenvalues:np.ndarray, replace_with=1.0) -> np.ndarray:
    """_summary_

    Args:
        eigenvalues (np.ndarray): _description_
        replace_with (float, optional): _description_. Defaults to 1.0.

    Returns:
        np.ndarray: _description_
    """    
    neg_indices = np.argwhere([eval < 0 for eval in eigenvalues])

    if len(neg_indices) > 1:
        print("more than one neg. eigenvalue: "+str(index_to_replace))
        index_to_replace = np.argmin(eigenvalues)
    else :
        index_to_replace = neg_indices[0]
    
    replaced_eigenvalues = eigenvalues
    eigenvalues[index_to_replace] = replace_with #TODO: MF determine if we stick to this method, what it depends on, etc

    return replaced_eigenvalues

# endregion Generalized

# region Hessian-specific

def reform_hessian(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        eigenvalues (np.ndarray[float]): _description_
        eigenvectors (np.ndarray[float]): _description_

    Returns:
        np.ndarray: _description_
    """
    reformed_hessian = (np.diag(eigenvalues).dot(np.transpose(eigenvectors)))
    return reformed_hessian

def invert_ts_curvature(hessian_matrix: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        hessian_matrix (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """    
    eigenvalues, eigenvectors = decompose(hessian_matrix)
    inv_curv_hessian = reform_hessian(replace_neg_eigenvalue(eigenvalues), eigenvectors)
    
    return inv_curv_hessian



# endregion Hessian-specific