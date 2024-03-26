"""_summary_

"""
from __future__ import division, print_function, absolute_import
import copy

import numpy as np
import constants as co

# region Generalized


def decompose(matrix: np.ndarray) -> (np.ndarray, np.ndarray):
    """_summary_

    Args:
        matrix (np.ndarray): Matrix to decompose, matrix must be square.

    Returns:
        (np.ndarray, np.ndarray): (eigenvalues, eigenvectors) where eigenvalues
         is of shape (1,n) and eigenvectors is of shape (n,n) with n rows of
         eigenvectors of length n.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvalues, eigenvectors


def replace_neg_eigenvalue(
    eigenvalues: np.ndarray, replace_with=1.0, zer_out_neg=False, units=co.KJMOLA
) -> np.ndarray:
    """_summary_

    Args:
        eigenvalues (np.ndarray): _description_
        replace_with (float, optional): _description_. Defaults to 1.0.

    Returns:
        np.ndarray: _description_
    """
    neg_indices = np.argwhere([eval < 0 for eval in eigenvalues])

    if len(neg_indices) > 1:
        print("more than one neg. eigenvalue: " + str(neg_indices))
        index_to_replace = np.argmin(eigenvalues)
    else:
        index_to_replace = neg_indices[0][0]
    replaced_eigenvalues = copy.deepcopy(eigenvalues)

    # TODO: MF - Discussed with PO, decide if this should be implemented as it is not in current Q2MM
    # for neg_index in neg_indices:
    #     if neg_index != index_to_replace:
    #         replaced_eigenvalues[neg_index] = 0
    if zer_out_neg:
        for neg_index in neg_indices:
            replaced_eigenvalues[neg_index[0]] = 0.00
    replaced_eigenvalues[
        index_to_replace
    ] = replace_with * co.HESSIAN_CONVERSION  if units == co.KJMOLA else replace_with # TODO: MF determine if we stick to this method, what it depends on, etc

    return replaced_eigenvalues


# endregion Generalized

# region Hessian-specific (Hermitian)


def reform_hessian(eigenvalues: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        eigenvalues (np.ndarray[float]): _description_
        eigenvectors (np.ndarray[float]): _description_

    Returns:
        np.ndarray: _description_
    """
    reformed_hessian = eigenvectors.dot(np.diag(eigenvalues).dot(eigenvectors.T))
    return reformed_hessian


def invert_ts_curvature(hessian_matrix: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        hessian_matrix (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    eigenvalues, eigenvectors = decompose(hessian_matrix)
    inv_curv_hessian = reform_hessian(
        replace_neg_eigenvalue(eigenvalues, zer_out_neg=True), eigenvectors
    )

    #check_evals = np.diag()

    if not inv_curv_hessian.all() >= 0.0:
        print("Inverted Hessian has negative values...")
        print(str(sum(inv_curv_hessian > 0))+" negative values...")

    return inv_curv_hessian


# endregion Hessian-specific
