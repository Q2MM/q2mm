"""
This module contains tools to estimate force field parameters with the
Seminario method.
"""
from __future__ import division, print_function, absolute_import

import numpy as np
import parmed

__all__ = ['make_angled_ff', 'make_bonded_ff', 'seminario_angle',
            'seminario_bond', 'seminario_sum', 'sub_hessian']

def sub_hessian(hessian, atom1, atom2) :
    """
    Subsample the Hessian matrix that is formed between atom1 and atom2
    as well as calculating the vector from atom1 to atom2

    Parameters
    ----------
    hessian : numpy.ndarray
        the Hessian matrix
    atom1 : parmed.Atom
        the first atom
    atom2 : parmed.Atom
        the second atom

    Returns
    -------
    numpy.ndarray
        the vector from atom1 to atom2
    numpy.ndarray
        the eigenvalues of the submatrix
    numpy.ndarray
        the eigenvector of the submatrix
    """
    vec12 = np.asarray([atom1.xx - atom2.xx, atom1.xy - atom2.xy, atom1.xz - atom2.xz])
    vec12 = vec12 / np.linalg.norm(vec12)

    submat = -hessian[3*atom1.idx:3*atom1.idx+3, 3*atom2.idx:3*atom2.idx+3]
    eigval, eigvec = np.linalg.eig(submat)
    return vec12, eigval, eigvec

def seminario_sum(vec, eigval, eigvec) :
    """
    Average the projections of the eigenvector on a specific unit vector
    according to Seminario

    Parameters
    ----------
    vec : numpy.ndarray
        the unit vector
    eigval : numpy.ndarray
        the eigenvalues of a Hessian submatrix
    eigvec : numpy.ndarray
        the eigenvectors of a Hessian submatrix

    Returns
    -------
    float :
        the average projection
    """
    ssum = 0.0
    for i in range(3):
        ssum += eigval[i] * np.abs(np.dot(eigvec[:,i], vec))
    return ssum

def seminario_bond(bond, hessian, scaling) :
    """
    Estimate the bond force constant using the Seminario method, i.e. by
    analysing the Hessian submatrix. Will average over atom1->atom2 and
    atom2->atom1 force constants

    Parameters
    ----------
    bond : parmed.Bond
        the bond to estimate the force constant for
    hessian : numpy.ndarray
        the Hessian matrix
    scaling : float
        the Hessian scaling factor
    """

    vec12, eigval12, eigvec12 = sub_hessian(hessian, bond.atom1, bond.atom2)
    f12 = seminario_sum(vec12, eigval12, eigvec12)

    vec21, eigval21, eigvec21 = sub_hessian(hessian, bond.atom2, bond.atom1)
    f21 = seminario_sum(vec21, eigval21, eigvec21)

    # 2240.87 is from Hartree/Bohr ^2 to kcal/mol/A^2
    # 418.4 is kcal/mol/A^2 to kJ/mol/nm^2
    return scaling * 2240.87 * 418.4 * 0.5 * (f12+f21)

def make_bonded_ff(struct, xyz_orig, hess, bonds, scaling) :
    """
    Make bonded force field parameters for selected bonds using the Seminario
    method.

    This will print the force field parameters to standard output

    Parameters
    ----------
    struct : parmed.Structure
        the structure of the model, including optimized coordinates
    xyz_orig : numpy.ndarray
        the original xyz coordinates
    hess : numpy.ndarray
        the Hessian matrix
    bonds : list of strings
        the bond definitions
    scaling : float
        the scaling factor for the Hessian
    """

    for bond in bonds :
        sel1, sel2 = bond.strip().split("-")
        atom1 = struct.view[sel1].atoms[0]
        atom2 = struct.view[sel2].atoms[0]
        bond = parmed.Bond(atom1, atom2)
        struct.bonds.append(bond)
    struct_orig = struct.copy(type(struct))
    struct_orig.coordinates = xyz_orig

    print("Bond\tidx1\tidx2\tr(x-ray)\tr(opt) [nm]\tk [kJ/mol/nm2]")
    for bond, bond_orig, bondmask in zip(struct.bonds, struct_orig.bonds, bonds) :
        force = seminario_bond(bond, hess, scaling)
        print("%s\t%d\t%d\t%.4f\t%.4f\t%.4f"%(bondmask, bond.atom1.idx+1,
            bond.atom2.idx+1, bond_orig.measure()*0.1, bond.measure()*0.1, force))

def seminario_angle(angle, hessian, scaling) :
    """
    Estimate the angle force constant using the Seminario method, i.e. by
    analysing the Hessian submatrix.

    Parameters
    ----------
    angle : parmed.Angle
        the angle to estimate the force constant for
    hessian : numpy.ndarray
        the Hessian matrix
    scaling : float
        the Hessian scaling factor
    """

    vec12, eigval12, eigvec12 = sub_hessian(hessian, angle.atom1, angle.atom2)
    vec32, eigval32, eigvec32 = sub_hessian(hessian, angle.atom3, angle.atom2)

    un = np.cross(vec32, vec12)
    un = un / np.linalg.norm(un)
    upa = np.cross(un, vec12)
    upc = np.cross(vec32, un)

    sum1 = seminario_sum(upa, eigval12, eigvec12)
    sum2 = seminario_sum(upc, eigval32, eigvec32)

    bond12 = parmed.Bond(angle.atom1, angle.atom2)
    len12 = bond12.measure()
    bond32 = parmed.Bond(angle.atom3, angle.atom2)
    len32 = bond32.measure()

    f = 1.0 / (1.0/(sum1*len12*len12)
              +1.0/(sum2*len32*len32))

    # 627.5095 is Hatree to kcal/mol
    # 4.184 is kcal/mol to kJ/mol
    return scaling * 627.5095 * 4.184 * f

def make_angled_ff(struct, xyz_orig, hess, angles, scaling) :
    """
    Make angle force field parameters for selected angles using the Seminario
    method.

    This will print the force field parameters to standard output

    Parameters
    ----------
    struct : parmed.Structure
        the structure of the model, including optimized coordinates
    xyz_orig : numpy.ndarray
        the original xyz coordinates
    hess : numpy.ndarray
        the Hessian matrix
    angles : list of strings
        the angle definitions
    scaling : float
        the scaling factor for the Hessian
    """

    for angle in angles :
        sel1, sel2, sel3 = angle.strip().split("-")
        atom1 = struct.view[sel1].atoms[0]
        atom2 = struct.view[sel2].atoms[0]
        atom3 = struct.view[sel3].atoms[0]
        angle = parmed.Angle(atom1, atom2, atom3)
        struct.angles.append(angle)
    struct_orig = struct.copy(type(struct))
    struct_orig.coordinates = xyz_orig

    print("Angle\tidx1\tidx2\tidx3\ttheta(x-ray)\ttheta(opt)\tk [kJ/mol]")
    for angle, angle_orig, anglemask in zip(struct.angles, struct_orig.angles, angles) :
        force = seminario_angle(angle, hess, scaling)
        print("%s\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f"%(anglemask, angle.atom1.idx+1,
            angle.atom2.idx+1, angle.atom3.idx+1,
            angle_orig.measure(), angle.measure(), force))
