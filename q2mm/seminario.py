"""
Estimates bond, angle force constants  and outputs to a new forcefield with the
Seminario method. This code is used for both AMBER and MM3 FF generation/preparation,
but it should only be called as a script for AMBER frcmod generation from a mol2 file
and a Gausian log file or an fchk file.  

Seminario estimation of force constants should be done at the same time as bond length
and angle equilibrium value averaging via -av in parameters.py.

Code adapted from Samuel Genheden and can be found at
https://github.com/SGenheden/Seminario
"""
from __future__ import division, print_function, absolute_import
import argparse
from collections import Counter
import os
import sys

import numpy as np
import parmed

from linear_algebra import invert_ts_curvature
from datatypes import MM3, AmberFF

import logging
import logging.config
import constants as co
from filetypes import GaussLog
import utilities


__all__ = [
    "make_angled_ff",
    "make_bonded_ff",
    "seminario_angle",
    "seminario_bond",
    "seminario_sum",
    "sub_hessian",
]
__all__ += ["parse_fchk"]

# region forcefield


def sub_hessian(hessian, atom1, atom2):
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

    submat = -hessian[
        3 * atom1.idx : 3 * atom1.idx + 3, 3 * atom2.idx : 3 * atom2.idx + 3
    ]
    eigval, eigvec = np.linalg.eig(submat)
    return vec12, eigval, eigvec

def create_unit_vector(atom1, atom2):
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
    vec21 = -vec12
    #eigval, eigvec = np.linalg.eig(submat)
    return vec12

def get_subhessian(hessian, atom1, atom2):
    submat_11 = -hessian[
        3 * atom1.idx : 3 * atom1.idx + 3, 3 * atom1.idx : 3 * atom1.idx + 3
    ]
    submat_12 = -hessian[
        3 * atom1.idx : 3 * atom1.idx + 3, 3 * atom2.idx : 3 * atom2.idx + 3
    ]
    submat_21 = -hessian[
        3 * atom2.idx : 3 * atom2.idx + 3, 3 * atom1.idx : 3 * atom1.idx + 3
    ]
    submat_22 = -hessian[
        3 * atom2.idx : 3 * atom2.idx + 3, 3 * atom2.idx : 3 * atom2.idx + 3
    ]
    submat_1 = np.hstack((submat_11, submat_12))
    submat_2 = np.hstack((submat_21, submat_22))
    submat = np.vstack(submat_1, submat_2)

    return submat

def create_unit_vectors(atom1, atom2, atom3):
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

    submat = -hessian[
        3 * atom1.idx : 3 * atom1.idx + 3, 3 * atom2.idx : 3 * atom2.idx + 3
    ]
    eigval, eigvec = np.linalg.eig(submat)
    return vec12, eigval, eigvec


def sub_hessian_new(hessian, atom1, atom2):
    """
    Subsample the Hessian matrix that is formed between atom1 and atom2
    as well as calculating the vector from atom1 to atom2

    Parameters
    ----------
    hessian : numpy.ndarray
        the Hessian matrix
    atom1 : Atom
        the first atom
    atom2 : Atom
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
    vec12 = np.asarray([atom1.x - atom2.x, atom1.y - atom2.y, atom1.z - atom2.z])
    vec12 = vec12 / np.linalg.norm(vec12)

    submat = -hessian[
        3 * atom1.index : 3 * atom1.index + 3, 3 * atom2.index : 3 * atom2.index + 3
    ]
    eigval, eigvec = np.linalg.eig(submat)
    return vec12, eigval, eigvec

def po_sum(unit_vec, eigval, eigvec):
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
    for i in range(len(eigval)):
        ssum += eigval[i] * np.abs(np.dot(eigvec[:, i], unit_vec))
    return ssum


def seminario_sum(vec, eigval, eigvec):
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
        ssum += eigval[i] * np.abs(np.dot(eigvec[:, i], vec))
    return ssum

def po_bond(bond, hessian, scaling=0.963, convert=False):
    unit_vector_ab = create_unit_vector(bond.atom1, bond.atom2)
    subhessian = get_subhessian(hessian, bond.atom1, bond.atom2)
    eigval, eigvec = np.linalg.eig(subhessian)

    ab_sum = po_sum(unit_vector_ab, eigval, eigvec)


    # 2240.87 is from Hartree/Bohr ^2 to kcal/mol/A^2
    # 418.4 is kcal/mol/A^2 to kJ/mol/nm^2

    if convert:
        return scaling * 2240.87 * 418.4 * 0.5 * po_sum
    else:
        return scaling * 0.5 * po_sum

def seminario_bond(bond, hessian, scaling=0.963, convert=False):
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
    print("vec12: " + str(vec12))
    print("eigval12: " + str(eigval12))
    print("eigvec12: " + str(eigvec12))
    f12 = seminario_sum(vec12, eigval12, eigvec12)
    print("f12: " + str(f12))

    vec21, eigval21, eigvec21 = sub_hessian(hessian, bond.atom2, bond.atom1)
    print("vec21: " + str(vec21))
    print("eigval21: " + str(eigval21))
    print("eigvec21: " + str(eigvec21))
    f21 = seminario_sum(vec21, eigval21, eigvec21)
    print("f21: " + str(f21))

    # 2240.87 is from Hartree/Bohr ^2 to kcal/mol/A^2
    # 418.4 is kcal/mol/A^2 to kJ/mol/nm^2

    if convert:
        return scaling * 2240.87 * 418.4 * 0.5 * (f12 + f21)
    else:
        return scaling * 0.5 * (f12 + f21)


def seminario_bond_new(atom1, atom2, hessian, scaling=0.963):
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

    vec12, eigval12, eigvec12 = sub_hessian_new(hessian, atom1, atom2)
    f12 = seminario_sum(vec12, eigval12, eigvec12)

    vec21, eigval21, eigvec21 = sub_hessian_new(hessian, atom2, atom1)
    f21 = seminario_sum(vec21, eigval21, eigvec21)

    # 2240.87 is from Hartree/Bohr ^2 to kcal/mol/A^2
    # 418.4 is kcal/mol/A^2 to kJ/mol/nm^2
    return scaling * 2240.87 * 418.4 * 0.5 * (f12 + f21)


def make_bonded_ff(struct, xyz_orig, hess, bonds, scaling=0.963):
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

    for bond in bonds:
        sel1, sel2 = bond.strip().split("-")
        atom1 = struct.view[sel1].atoms[0]
        atom2 = struct.view[sel2].atoms[0]
        bond = parmed.Bond(atom1, atom2)
        struct.bonds.append(bond)
    struct_orig = struct.copy(type(struct))
    struct_orig.coordinates = xyz_orig

    print("Bond\tidx1\tidx2\tr(x-ray)\tr(opt) [nm]\tk [kJ/mol/nm2]")
    for bond, bond_orig, bondmask in zip(struct.bonds, struct_orig.bonds, bonds):
        force = seminario_bond(bond, hess, scaling)
        print(
            "%s\t%d\t%d\t%.4f\t%.4f\t%.4f"
            % (
                bondmask,
                bond.atom1.idx + 1,
                bond.atom2.idx + 1,
                bond_orig.measure() * 0.1,
                bond.measure() * 0.1,
                force,
            )
        )


def seminario_angle(angle, hessian, scaling=0.963, convert=False):
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

    f = 1.0 / (1.0 / (sum1 * len12 * len12) + 1.0 / (sum2 * len32 * len32))

    # 627.5095 is Hatree to kcal/mol
    # 4.184 is kcal/mol to kJ/mol
    if convert:
        return scaling * 627.5095 * 4.184 * f
    else:
        return scaling * f


def make_angled_ff(struct, xyz_orig, hess, angles, scaling=0.963):
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

    for angle in angles:
        sel1, sel2, sel3 = angle.strip().split("-")
        atom1 = struct.view[sel1].atoms[0]
        atom2 = struct.view[sel2].atoms[0]
        atom3 = struct.view[sel3].atoms[0]
        angle = parmed.Angle(atom1, atom2, atom3)
        struct.angles.append(angle)
    struct_orig = struct.copy(type(struct))
    struct_orig.coordinates = xyz_orig

    print("Angle\tidx1\tidx2\tidx3\ttheta(x-ray)\ttheta(opt)\tk [kJ/mol]")
    for angle, angle_orig, anglemask in zip(struct.angles, struct_orig.angles, angles):
        force = seminario_angle(angle, hess, scaling)
        print(
            "%s\t%d\t%d\t%d\t%.4f\t%.4f\t%.4f"
            % (
                anglemask,
                angle.atom1.idx + 1,
                angle.atom2.idx + 1,
                angle.atom3.idx + 1,
                angle_orig.measure(),
                angle.measure(),
                force,
            )
        )


# endregion

# region Gaussian


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
        while line and not line.startswith(startline):
            line = f.readline()
        while line and not line.startswith(endline):
            line = f.readline()
            if not line.startswith(endline):
                arr.extend(line.strip().split())
        return np.array(arr, dtype=float)

    crds = None
    hess = None
    with open(filename, "r") as f:
        # First the coordinates
        crds = _parse_array(f, "Current cartesian coordinates", "Force Field")
        # Then the Hessian
        hess = _parse_array(f, "Cartesian Force Constants", "Dipole Moment")

    # Make the Hessian in square form
    i = 0
    n = len(crds)
    hess_sqr = np.zeros([n, n])
    for j in range(n):
        for k in range(j + 1):
            hess_sqr[j, k] = hess[i]
            hess_sqr[k, j] = hess[i]
            i += 1

    natoms = int(n / 3)
    return crds.reshape([natoms, 3]), hess_sqr


# endregion

# region Arguments


def return_params_parser(add_help=True):
    """
    Returns an argparse.ArgumentParser object for the selection of
    parameters.
    """
    if add_help:
        description = (
            __doc__
            + """
PTYPES:
ae   - equilibrium angles
af   - angle force constants
be   - equilibrium bond lengths
bf   - bond force constants
df   - dihedral force constants
imp1 - improper torsions (1st MM3* column)
imp2 - improper torsions (2nd MM3* column)
sb   - stretch-bend force constants
q    - bond dipoles
vdwe - van der Waals epsilon
vdwr - van der Waals radius"""
        )
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawTextHelpFormatter, description=description
        )
    else:
        parser = argparse.ArgumentParser(add_help=False)
    par_group = parser.add_argument_group("seminario")
    par_group.add_argument(
        "-o",
        "--ff-out",
        type=str,
        metavar="filename.seminario.frcmod",
        default="amber.seminario.frcmod",
        help=(
            "Use mol2 file and Gaussian Hessian to generate a new force field where\n"
            "each force constant value is replaced\n"
            "by its value estimated from the seminario calculation\n"
            "of force constants in the structure."
        ),
    )
    par_group.add_argument(
        "-i",
        "--ff-in",
        metavar="filename.frcmod",
        default="amber.frcmod",
        help="Path to input frcmod.",
    )
    par_group.add_argument(
        "--mol",
        "-m",
        type=str,
        metavar="structure.mol2",
        default=None,
        help="Read this mol2 file, units are in Angstrom.",
    )
    par_group.add_argument(
        "--pdb",
        type=str,
        metavar="structure.pdb",
        default=None,
        help="Read this pdb file, units are in Angstrom.",
    )
    par_group.add_argument(
        "--log",
        "-gl",
        type=str,
        metavar="gaussian.log",
        default=None,
        help="Gaussian Hessian is extracted from this .log file for seminario calculations. Units are in Bohr.",
    )
    par_group.add_argument(
        "--fchk",
        "-gf",
        type=str,
        metavar="gaussian.fchk",
        default=None,
        help="Gaussian Hessian and structure are extracted from this .fchk file for seminario calculations. Units are in Bohr.",
    )
    par_group.add_argument(
        "--params",
        "-p",
        type=str,
        metavar="parameters.txt",
        default=None,
        help="Text file containing the parameters (bonds, angles) to be calculated.",
    )
    par_group.add_argument(
        "--mm3",
        type=str,
        default=False,
        help="Flag indicating that the force field type is MM3, used only for testing.",
    )
    return parser


# endregion

# region Stand-alone AMBER Seminario methods process
print("pre main")


def main(args):
    print("in main")
    if sys.version_info > (3, 0):
        if isinstance(args, str):
            args = args.split()
    else:
        if isinstance(args, basestring):
            args = args.split()
    parser = return_params_parser()
    args = parser.parse_args()

    print(str(args))

    assert args.ff_in, "Input frcmod AMBER FF file is required!"

    assert (args.mol or args.pdb) and (
        args.log or args.fchk
    ), "Both a mol2 structure file and a Gaussian log or Gaussian fchk (DFT Hessian) file are needed!"

    if args.mm3:
        ff_in = MM3(args.ff_in)
        ff_in.import_ff()
        print("mm3 ff imported")
    else:
        ff_in = AmberFF(args.ff_in)
        ff_in.import_ff()
        print("amber ff imported")

    params = []
    if args.params is None:
        params = ff_in.params
    else:
        with open(args.params, "r") as param_file:
            lines = param_file.readlines()
            for line in lines:
                params.append(line)

    if args.fchk:
        dft_coords, dft_hessian = parse_fchk(args.fchk)
        print("fchk parsed")
    elif args.log:
        log = GaussLog(args.log)
        print("glog object")
        structures = log.structures
        print("got glog structures")
        # TODO: get coords by pulling from each atom
        dft_coords = np.array(log.structures[-1].coords)
        dft_hessian = log.structures[-1].hess

    min_hessian = invert_ts_curvature(dft_hessian)
    print("hessian curvature inverted: " + str(min_hessian))

    struct = (
        parmed.load_file(args.mol, structure=True)
        if args.mol
        else parmed.load_file(args.pdb, structure=True)
    )
    mol_coords = np.array(struct.coordinates)
    # struct.coordinates is type(np.array) of shape n_atoms, 3
    struct.coordinates = dft_coords  # *0.529177249 # 0.529177249 is Bohr to A #TODO: MF - check which units Q2MM uses

    print("structure: " + str(struct))

    temp_struct = struct.copy(type(struct))
    temp_struct.bonds.clear()
    temp_struct.angles.clear()
    print(
        "will now iterate across params to find matching structure and calculate new param value."
    )
    for param in params:
        print(str(param))
        print(param.ptype)
        if param.ptype is "bf":
            print("bf types: " + str(param.atom_types))
            for bond in struct.bonds:
                if utilities.is_same_bond(
                    param.atom_types,
                    utilities.convert_atom_type_pair(
                        [bond.atom1.type, bond.atom2.type]
                    ),
                ):
                    print("matched")
                    s_bond = seminario_bond(bond, min_hessian, convert=args.fchk)
                    print("seminario bond: "+str(s_bond))
                    p_bond = po_bond(bond, min_hessian, convert=args.fchk)
                    print("po bond: "+str(p_bond))
                    param.value = p_bond
                    print("new param value: " + str(param.value))
        if param.ptype is "be":
            print("be")
            for bond in struct.bonds:
                possible_matches = [
                    [bond.atom1.type, bond.atom2.type],
                    [bond.atom2.type, bond.atom1.type],
                ]
                if param.atom_types in possible_matches:
                    temp_struct.bonds.append(bond)
                    param.value = bond.measure()
        if param.ptype is "af":
            print("af")
            for angle in struct.angles:
                possible_matches = [
                    [angle.atom1.type, angle.atom2.type, angle.atom3.type],
                    [angle.atom3.type, angle.atom2.type, angle.atom1.type],
                ]
                if param.atom_types in possible_matches:
                    param.value = seminario_angle(angle, min_hessian, convert=args.fchk)
        if param.ptype is "ae":
            print("ae")
            for angle in struct.angles:
                possible_matches = [
                    [angle.atom1.type, angle.atom2.type, angle.atom3.type],
                    [angle.atom3.type, angle.atom2.type, angle.atom1.type],
                ]
                if param.atom_types in possible_matches:
                    temp_struct.angles.append(angle)
                    param.value = (
                        angle.measure()
                    )  # TODO: MF - CHECK UNITS, does setting coords in struct set them for atoms? NOPE
        # NOTE: user must make sure mol2 structure is the same as gaussian log or fchk structure (just in IRC)

        struct = temp_struct

    make_bonded_ff(struct, mol_coords, dft_hessian, struct.bonds)

    make_angled_ff(struct, mol_coords, dft_hessian, struct.angles)

    make_bonded_ff(struct, mol_coords, min_hessian, struct.bonds)

    make_angled_ff(struct, mol_coords, min_hessian, struct.angles)

    # Write out new frcmod
    ff_in.export_ff(args.ff_out, params)


# endregion

if __name__ == "__main__":
    logging.config.dictConfig(co.LOG_SETTINGS)
    main(sys.argv[1:])

# region tools


def seminario_ff():
    """
    The entry point for the Seminario script.

    Setting up argument parser, load files and then estimate force field.
    """

    argparser = argparse.ArgumentParser(
        description="Script to compute bond force constant with Seminario"
    )
    argparser.add_argument("-f", "--checkpoint", help="the formated checkpoint file")
    argparser.add_argument("-s", "--struct", help="a structure file")
    argparser.add_argument("-b", "--bonds", nargs="+", help="the bonds")
    argparser.add_argument("-a", "--angles", nargs="+", help="the angles")
    argparser.add_argument(
        "--scaling", type=float, help="the frequency scaling factor", default=0.963
    )
    argparser.add_argument(
        "--saveopt",
        action="store_true",
        help="save the optimized coordinates to file",
        default=False,
    )
    args = argparser.parse_args()

    if args.checkpoint is None or args.struct is None:
        print("Nothing to be done. Use -h to see help. \n Exit.")
        return

    struct = parmed.load_file(args.struct, skip_bonds=True)
    xyz_opt, hess = parse_fchk(args.checkpoint)
    xyz_orig = np.array(struct.coordinates)
    # struct.coordinates is type(np.array) of shape n_atoms, 3
    struct.coordinates = xyz_opt * 0.529177249  # 0.529177249 is Bohr to A

    if args.saveopt:
        base, ext = os.path.splitext(args.struct)
        struct.save(base + "_opt" + ext)

    if args.bonds is not None:
        make_bonded_ff(struct, xyz_orig, hess, args.bonds, args.scaling)

    if args.angles is not None:
        make_angled_ff(struct, xyz_orig, hess, args.angles, args.scaling)


# endregion
