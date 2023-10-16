from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from abc import abstractmethod

import logging
from typing import List
import numpy as np
import os
import re
import sys

import constants as co
import utilities

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

# Print out full matrices rather than having Numpy truncate them.
# np.nan seems to no longer be supported for untruncated printing
# of arrays. The suggestion is to use sys.maxsize but I haven't checked
# that this works for python2 so leaving the commented code for now.
# np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=sys.maxsize)


class Atom(object):
    """
    Data class for a single atom.
    """

    __slots__ = [
        "atom_type",
        "atom_type_name",
        "atomic_num",
        "atomic_mass",
        "bonded_atom_indices",
        "coords_type",
        "_element",
        "_exact_mass",
        "index",
        "partial_charge",
        "x",
        "y",
        "z",
        "props",
    ]

    def __init__(
        self,
        atom_type: str = None,
        atom_type_name: str = None,
        atomic_num: int = None,
        atomic_mass: float = None,
        bonded_atom_indices=None,
        coords=None,
        coords_type=None,
        element: str = None,
        exact_mass=None,
        index: int = None,
        partial_charge: float = None,
        x: float = None,
        y: float = None,
        z: float = None,
    ):
        """Atom object containing relevant properties and metadata.

        Units: Angstrom

        Note:
            TODO Values are all optional because of the currently established q2mm code, however this
            is bad practice, any strictly necessary properties should be required arguments, such as atom type, index, and
            coordinates.

        Args:
            atom_type (str, optional): The integer atom type according to atom.typ file. Defaults to None.
            atom_type_name (str, optional): The name of the atom type corresponding to the integer atom type in the atom.typ file. Defaults to None.
            atomic_num (int, optional): Atomic number (element number) of the atom. Defaults to None.
            atomic_mass (float, optional): TODO. Defaults to None.
            bonded_atom_indices (TODO, optional): TODO. Defaults to None.
            coords (TODO maybe np.ndarray, optional): Atom coordinates. Defaults to None.
            coords_type (TODO, optional): TODO Is this even ever used?. Defaults to None.
            element (str, optional): The atom element (e.g. C, N, O, H). Defaults to None.
            exact_mass (_type_, optional): TODO. Defaults to None.
            index (int, optional): The index number of the atom in its original structural file. Defaults to None.
            partial_charge (float, optional): TODO is this even really used?. Defaults to None.
            x (float, optional): X coordinate of the atom. Defaults to None.
            y (float, optional): Y coordinate of the atom. Defaults to None.
            z (float, optional): Z coordinate of the atom. Defaults to None.
        """
        self.atom_type = atom_type
        self.atom_type_name = atom_type_name
        self.atomic_num = atomic_num  # This is the atom index in the original structure file, 1-based NOT 0-based
        self.atomic_mass = atomic_mass
        self.bonded_atom_indices = bonded_atom_indices
        self.coords_type = coords_type
        self._element = element
        self._exact_mass = exact_mass
        self.index = index
        self.partial_charge = partial_charge
        self.x = x
        self.y = y
        self.z = z
        if coords:  # coordinates are all in Angstroms and Cartesian
            self.x = float(coords[0])
            self.y = float(coords[1])
            self.z = float(coords[2])
        self.props = {}

    def __repr__(self):
        return "{}[{},{},{}]".format(self.atom_type_name, self.x, self.y, self.z)

    @property
    def coords(self) -> np.ndarray:
        """Getter method for coords property.

        Returns:
            np.ndarray: Array of Cartesian coordinates of atom of form [x, y, z].
        """
        return np.array([self.x, self.y, self.z])

    @coords.setter
    def coords(self, value):
        """Setter method for coords property.

        Args:
            value (TODO): Cartesian coordinates of atom of form [x, y, z]
        """
        try:
            self.x = value[0]
            self.y = value[1]
            self.z = value[2]
        except TypeError:
            pass

    @property
    def element(self):
        if self._element is None:
            self._element = co.MASSES.items()[self.atomic_num - 1][0]
        return self._element

    @element.setter
    def element(self, value):
        self._element = value

    @property
    def exact_mass(self):
        if self._exact_mass is None:
            self._exact_mass = co.MASSES[self.element]
        return self._exact_mass

    @exact_mass.setter
    def exact_mass(self, value):
        self._exact_mass = value


class DOF(object):
    """
    Abstract data class for a single degree of freedom.
    """

    __slots__ = ["atom_nums", "comment", "value", "ff_row"]

    def __init__(
        self,
        atom_nums: List[int] = None,
        comment: str = None,
        value: float = None,
        ff_row: int = None,
    ):
        """Abstract Class for a Degree Of Freedom (DOF) containing the bare bones properties necessary.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the DOF. Defaults to None.
            comment (str, optional): Any comment associated with the DOF TODO Is this used?. Defaults to None.
            value (float, optional): Value of the DOF. Defaults to None.
            ff_row (int, optional): Row of the FF which models the DOF. Defaults to None.
        """
        self.atom_nums: List[int] = atom_nums
        """ TODO atom_indices is a more intuitive name, 
        but use of this property is too widespread (with poor referencing) to change atm,
        refactor this name when there is time."""
        self.comment = comment
        self.value = value
        self.ff_row = ff_row

    def __repr__(self):
        return "{}[{}]({})".format(
            self.__class__.__name__, "-".join(map(str, self.atom_nums)), self.value
        )

    def as_data(self, **kwargs):
        # Sort of silly to have all this stuff about angles and
        # torsions in here, but they both inherit from this class.
        # I suppose it'd make more sense to create a structural
        # element class that these all inherit from.
        # Warning that I recently changed these labels, and that
        # may have consequences.
        if self.__class__.__name__.lower() == "bond":
            typ = "b"
        elif self.__class__.__name__.lower() == "angle":
            typ = "a"
        elif self.__class__.__name__.lower() == "torsion":
            typ = "t"
        datum = Datum(val=self.value, typ=typ, ff_row=self.ff_row)
        for i, atom_num in enumerate(self.atom_nums):
            setattr(datum, "atm_{}".format(i + 1), atom_num)
        for k, v in kwargs.items():
            setattr(datum, k, v)
        return datum

    def is_same_DOF(self, other) -> bool:
        """Comparison operator for DOFs. Returns true if the DOFs are identical
        based on the indices of the atoms involved. Relies on the assumption that
        atom indices are not different from structure to structure, TODO this is a
        fallacy which should be addressed at some point or at least emphasized to
        the user in documentation.

        Args:
            other (DOF): The DOF to which to compare self.

        Returns:
            bool: True if DOF is identical to self, else False.
        """
        assert other is DOF
        return all(self.atom_nums == other.atom_nums) or all(
            reversed(self.atom_nums) == other.atom_nums
        )


class Bond(DOF):
    """
    Data class for a single bond.
    """

    __slots__ = ["atom_nums", "comment", "value", "ff_row", "order"]

    def __init__(
        self,
        atom_nums: List[int] = None,
        comment: str = None,
        value: float = None,
        ff_row: int = None,
        order: int = None,
    ):
        """Bond object containing the bare bones properties necessary.

        Note:
            As of yet, there is no need for this class to also track a force constant as it
            is used exclusively in the newer, schrodinger-independent code like seminario.py,
            but this could be a good place to store that data within a FF object if Param were
            to be replaced/condensed.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the bond of the form [index1, index2]. Defaults to None.
            comment (str, optional): Any comment associated with the Bond TODO Is this used?. Defaults to None.
            value (float, optional): Bond length in Angstrom. Defaults to None.
            ff_row (int, optional): Row of the FF which models the bond. Defaults to None.
            order (int, optional): Bond order (e.g. single bond - 1, double bond - 2...)
        """
        super(Bond, self).__init__(atom_nums, comment, value, ff_row)
        self.order = order


class Angle(DOF):
    """
    Data class for a single angle.
    """

    def __init__(self, atom_nums=None, comment=None, value=None, ff_row=None):
        """Angle object containing the bare bones properties necessary.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the Angle. Defaults to None.
            comment (str, optional): Any comment associated with the Angle TODO Is this used?. Defaults to None.
            value (float, optional): Value of the Angle in degrees. Defaults to None.
            ff_row (int, optional): Row of the FF which models the Angle. Defaults to None.
        """
        super(Angle, self).__init__(atom_nums, comment, value, ff_row)


class Torsion(DOF):
    """
    Data class for a single torsion.
    """

    def __init__(self, atom_nums=None, comment=None, value=None, ff_row=None):
        """Torsion/Dihedral object containing the bare bones properties necessary.

        Args:
            atom_nums (List[int], optional): Indices of the atoms involved in the torsion. Defaults to None.
            comment (str, optional): Any comment associated with the torsion TODO Is this used?. Defaults to None.
            value (float, optional): Value of the torsion angle in degrees. Defaults to None.
            ff_row (int, optional): Row of the FF which models the Torsion. Defaults to None.
        """
        super(Torsion, self).__init__(atom_nums, comment, value, ff_row)


class Structure(object):
    """
    Data for a single structure/conformer/snapshot.
    """

    __slots__ = ["_atoms", "_bonds", "_angles", "_torsions", "hess", "props", "origin_name"]

    def __init__(self, origin_name:str):
        # TODO: This should really be a constructor which accepts a bare minimum of
        # these fields and the rest are optional defaulted to None, good for error-protection
        # and just generally cleaner, more intuitive. An empty structure is never itself used,
        # so why have it as an option which simply complicates error-checking and tracking.
        self._atoms: List[Atom] = []
        self._bonds: List[Bond] = None
        self._angles: List[Angle] = None
        self._torsions: List[Torsion] = None
        self.hess = None
        self.props = {}
        self.origin_name:str = origin_name

    @property
    def coords(self):
        """
        Returns atomic coordinates as a list of lists.
        """
        return [atom.coords for atom in self._atoms]

    @property
    def atoms(self):
        if self._atoms == []:
            raise Exception(
                "structure._atoms is not defined, this must be done on creation."
            )
        return self._atoms

    @property
    def bonds(self):
        if self._bonds is None:
            raise Exception(
                "structure._bonds is not defined, this must be done on creation."
            )
        return self._bonds

    @property
    def angles(self):
        if self._angles is None:
            self._angles = self.identify_angles()
        return self._angles

    @property
    def torsions(self):
        if self._torsions is None:
            self._torsions = self.identify_torsions()
        return self._torsions

    # region Methods which ought to be refactored or might be unused but I'm too busy/scared to mess with yet

    def format_coords(self, format="latex", indices_use_charge=None):
        """
        Returns a list of strings/lines to easily generate coordinates
        in various formats.

        latex  - Makes a LaTeX table.
        gauss  - Makes output that matches Gaussian's .com filse.
        jaguar - Just like Gaussian, but include the atom number after the
                 element name in the left column.
        """
        # Formatted for LaTeX.
        if format == "latex":
            output = [
                "\\begin{tabular}{l S[table-format=3.6] "
                "S[table-format=3.6] S[table-format=3.6]}"
            ]
            for i, atom in enumerate(self._atoms):
                if atom.element is None:
                    ele = co.MASSES.items()[atom.atomic_num - 1][0]
                else:
                    ele = atom.element
                output.append(
                    "{0}{1} & {2:3.6f} & {3:3.6f} & "
                    "{4:3.6f}\\\\".format(ele, i + 1, atom.x, atom.y, atom.z)
                )
            output.append("\\end{tabular}")
            return output
        # Formatted for Gaussian .com's.
        elif format == "gauss":
            output = []
            for i, atom in enumerate(self._atoms):
                if atom.element is None:
                    ele = co.MASSES.items()[atom.atomic_num - 1][0]
                else:
                    ele = atom.element
                # Used only for a problem Eric experienced.
                # if ele == '': ele = 'Pd'
                if indices_use_charge:
                    if atom.index in indices_use_charge:
                        output.append(
                            " {0:s}--{1:.5f}{2:>16.6f}{3:16.6f}"
                            "{4:16.6f}".format(
                                ele, atom.partial_charge, atom.x, atom.y, atom.z
                            )
                        )
                    else:
                        output.append(
                            " {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}".format(
                                ele, atom.x, atom.y, atom.z
                            )
                        )
                else:
                    output.append(
                        " {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}".format(
                            ele, atom.x, atom.y, atom.z
                        )
                    )
            return output
        # Formatted for Jaguar.
        elif format == "jaguar":
            output = []
            for i, atom in enumerate(self._atoms):
                if atom.element is None:
                    ele = co.MASSES.items()[atom.atomic_num - 1][0]
                else:
                    ele = atom.element
                # Used only for a problem Eric experienced.
                # if ele == '': ele = 'Pd'
                label = "{}{}".format(ele, atom.index)
                output.append(
                    " {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}".format(
                        label, atom.x, atom.y, atom.z
                    )
                )
            return output

    def select_stuff(self, typ, com_match=None):
        """
        A much simpler version of select_data. It would be nice if select_data
        was a wrapper around this function.
        """
        stuff = []
        for thing in getattr(self, typ):
            if (
                com_match and any(x in thing.comment for x in com_match)
            ) or com_match is None:
                stuff.append(thing)
        return stuff

    def select_data(self, typ, com_match=None, **kwargs):
        """
        Selects bonds, angles, or torsions from the structure and returns them
        in the format used as data.

        typ       - 'bonds', 'angles', or 'torsions'.
        com_match - String or None. If None, just returns all of the selected
                    stuff (bonds, angles, or torsions). If a string, selects
                    only those that have this string in their comment.

                    In .mmo files, the comment corresponds to the substructures
                    name. This way, we only fit bonds, angles, and torsions that
                    directly depend on our parameters.
        """
        data = []
        logger.log(1, ">>> typ: {}".format(typ))
        for thing in getattr(self, typ):
            if (
                com_match and any(x in thing.comment for x in com_match)
            ) or com_match is None:
                datum = thing.as_data(**kwargs)
                # If it's a torsion we have problems.
                # Have to check whether an angle inside the torsion is near 0 or 180.
                if typ == "torsions":
                    atom_nums = [datum.atm_1, datum.atm_2, datum.atm_3, datum.atm_4]
                    angle_atoms_1 = [atom_nums[0], atom_nums[1], atom_nums[2]]
                    angle_atoms_2 = [atom_nums[1], atom_nums[2], atom_nums[3]]
                    for angle in self._angles:
                        if set(angle.atom_nums) == set(angle_atoms_1):
                            angle_1 = angle.value
                            break
                    for angle in self._angles:
                        if set(angle.atom_nums) == set(angle_atoms_2):
                            angle_2 = angle.value
                            break
                    try:
                        logger.log(1, ">>> atom_nums: {}".format(atom_nums))
                        logger.log(
                            1, ">>> angle_1: {} / angle_2: {}".format(angle_1, angle_2)
                        )
                    except UnboundLocalError:
                        logger.error(">>> atom_nums: {}".format(atom_nums))
                        logger.error(">>> angle_atoms_1: {}".format(angle_atoms_1))
                        logger.error(">>> angle_atoms_2: {}".format(angle_atoms_2))
                        if "angle_1" not in locals():
                            logger.error("Can't identify angle_1!")
                        else:
                            logger.error(">>> angle_1: {}".format(angle_1))
                        if "angle_2" not in locals():
                            logger.error("Can't identify angle_2!")
                        else:
                            logger.error(">>> angle_2: {}".format(angle_2))
                        logger.warning("WARNING: Using torsion anyway!")
                        data.append(datum)
                    if (
                        -20.0 < angle_1 < 20.0
                        or 160.0 < angle_1 < 200.0
                        or -20.0 < angle_2 < 20.0
                        or 160.0 < angle_2 < 200.0
                    ):
                        logger.log(
                            1, ">>> angle_1 or angle_2 is too close to 0 or 180!"
                        )
                        pass
                    else:
                        data.append(datum)
                    # atom_coords = [x.coords for x in atoms]
                    # tor_1 = geo_from_points(
                    #     atom_coords[0], atom_coords[1], atom_coords[2])
                    # tor_2 = geo_from_points(
                    #     atom_coords[1], atom_coords[2], atom_coords[3])
                    # logger.log(1, '>>> tor_1: {} / tor_2: {}'.format(
                    #     tor_1, tor_2))
                    # if -5. < tor_1 < 5. or 175. < tor_1 < 185. or \
                    #         -5. < tor_2 < 5. or 175. < tor_2 < 185.:
                    #     logger.log(
                    #         1,
                    #         '>>> tor_1 or tor_2 is too close to 0 or 180!')
                    #     pass
                    # else:
                    #     data.append(datum)
                else:
                    data.append(datum)
        assert data, "No data actually retrieved!"
        return data

    def get_aliph_hyds(self):
        """
        Returns the atom numbers of aliphatic hydrogens. These hydrogens
        are always assigned a partial charge of zero in MacroModel
        calculations.

        This should be subclassed into something is MM3* specific.
        """
        aliph_hyds = []
        for atom in self._atoms:
            if 40 < atom.atom_type < 49:
                for bonded_atom_index in atom.bonded_atom_indices:
                    bonded_atom = self._atoms[bonded_atom_index - 1]
                    if bonded_atom.atom_type == 3:
                        aliph_hyds.append(atom)
        logger.log(5, "  -- {} aliphatic hydrogen(s).".format(len(aliph_hyds)))
        return aliph_hyds

    def get_hyds(self):
        """
        Returns the atom numbers of any default MacroModel type hydrogens.

        This should be subclassed into something is MM3* specific.
        """
        hyds = []
        for atom in self._atoms:
            if 40 < atom.atom_type < 49:
                for bonded_atom_index in atom.bonded_atom_indices:
                    hyds.append(atom)
        logger.log(5, "  -- {} hydrogen(s).".format(len(hyds)))
        return hyds

    def get_dummy_atom_indices(self):
        """
        Returns a list of integers where each integer corresponds to an atom
        that is a dummy atom.

        Returns
        -------
        list of integers
        """
        dummies = []
        for atom in self._atoms:
            if atom.is_dummy:
                logger.log(10, "  -- Identified {} as a dummy atom.".format(atom))
                dummies.append(atom.index)
        return dummies

    # endregion

    def identify_angles(self) -> List[Angle]:
        """Returns angles identified and measured within self Structure.

        Note:
            TODO May need to add same logic of 0 vs 180 as in filetypes.py

        Returns:
            List[Angle]: angles in self Structure
        """
        angles: List[Angle] = []
        i = 0
        for a in self.bonds:
            i += 1
            for b in self.bonds[i:]:
                a1_index, a2_index = a.atom_nums
                b1_index, b2_index = b.atom_nums
                if a1_index == b1_index:
                    if a2_index != b2_index:
                        angle = utilities.measure_angle(
                            self.atoms[a2_index - 1].coords,
                            self.atoms[a1_index - 1].coords,
                            self.atoms[b2_index - 1].coords,
                        )
                        angles.append(
                            Angle(atom_nums=[a2_index, a1_index, b2_index], value=angle)
                        )
                if a1_index == b2_index:
                    if a2_index != b1_index:
                        angle = utilities.measure_angle(
                            self.atoms[a2_index - 1].coords,
                            self.atoms[a1_index - 1].coords,
                            self.atoms[b1_index - 1].coords,
                        )
                        angles.append(
                            Angle(atom_nums=[a2_index, a1_index, b1_index], value=angle)
                        )
                if a2_index == b2_index:
                    if a1_index != b1_index:
                        angle = utilities.measure_angle(
                            self.atoms[a1_index - 1].coords,
                            self.atoms[a2_index - 1].coords,
                            self.atoms[b1_index - 1].coords,
                        )
                        angles.append(
                            Angle(atom_nums=[a1_index, a2_index, b1_index], value=angle)
                        )
                if a2_index == b1_index:
                    if a1_index != b2_index:
                        angle = utilities.measure_angle(
                            self.atoms[a1_index - 1].coords,
                            self.atoms[a2_index - 1].coords,
                            self.atoms[b2_index - 1].coords,
                        )
                        angles.append(
                            Angle(atom_nums=[a1_index, a2_index, b2_index], value=angle)
                        )
        return angles

    def identify_torsions(self):  # TODO
        raise NotImplemented

    def get_atoms_in_DOF(self, dof: DOF) -> List[Atom]:
        """Returns a list of Atom objects which are involved in the DOF as implied by atom indices.

        Args:
            dof (DOF): Degree of Freedom (Bond, Angle, etc.) to query

        Returns:
            List[Atom]: Atom objects involved in the DOF.
        """
        return [self.atoms[idx - 1] for idx in dof.atom_nums]

    def get_DOF_atom_types_dict(self) -> dict:
        """Returns a dictionary of the atom types which correspond to each DOF in self.

        Returns:
            dict: dictionary of the form {DOF: [atom_type_name1, atom_type_name2, ...]}
        """
        dof_atom_type_dict = dict()
        for bond in self.bonds:
            dof_atom_type_dict[bond] = [
                atom.atom_type_name for atom in self.get_atoms_in_DOF(bond)
            ]
        for angle in self.angles:
            dof_atom_type_dict[angle] = [
                atom.atom_type_name for atom in self.get_atoms_in_DOF(angle)
            ]
        return dof_atom_type_dict


class File(object):
    """
    Base for every other filetype class. Identical to filetypes.py version,
    ported over for schrodinger independence in seminario.py
    """

    __slots__ = ["_lines", "path", "directory", "filename"]

    def __init__(self, path: str):
        """Instantiates a file object fro the file at the location path passed.

        Populates the directory and filename properties as well.

        Args:
            path (str): location of the file
        """
        self._lines = None
        self.path = os.path.abspath(path)
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)
        # self.name = os.path.splitext(self.filename)[0]

    @property
    def lines(self) -> List[str]:
        """Returns the lines of the file.

        Returns:
            List[str]: lines of the file
        """
        if self._lines is None:
            with open(self.path, "r") as f:
                self._lines = f.readlines()
        return self._lines

    def write(self, path, lines=None):
        """Writes lines to file at path.

        Args:
            path (str): location of file to write
            lines (List[str], optional): lines to write to file. Defaults to None, which then writes self.lines.
        """
        if lines is None:
            lines = self.lines
        with open(path, "w") as f:
            for line in lines:
                f.write(line)


class Mol2(File):
    """
    Used to retrieve structural data from mol2 files.

    Please ensure that mol2 atom types match the atom types specified in the force field.

    Note:
            Format for the data in the file can be found by searching
            Tripos Mol2 File Format SYBYL.
    """

    TRIPOS_FLAG = "@<TRIPOS>"
    MOLECULE_FLAG = "MOLECULE"
    ATOM_FLAG = "ATOM"
    BOND_FLAG = "BOND"

    __slots__ = ["_lines", "path", "directory", "filename", "_structures"]

    def __init__(self, path: str):
        """Creates a Mol2 object based on the path given, data is only structural.

        Args:
            path (str): Absolute path of mol2 file.
        """
        super(Mol2, self).__init__(path)
        self._structures: List[Structure] = None

    @property
    def structures(self) -> List[Structure]:
        """Returns the Structure objects extracted from the mol2 file at self.path.
        If None, indicating no extraction yet, parses the lines from the file to populate
        the structures list with Structures.

        Returns:
            List[Structure]: Structure objects extracted from parsing the mol2 file.
        """
        if self._structures is None:
            self.parse_lines()
        return self._structures

    def parse_lines(self):
        """Parses self.lines() as set by super to extract Structure objects to self.structures.

        It is safe to parse this with split because the mol2 format from SYBYL
         requires consistent data ordering matching the standard, otherwise the
         file is not in valid mol2 format.
        """
        # TODO this could be amended to use regular expression matching (regex) if slow
        self._structures: List[Structure] = []
        joined_lines = "".join(self.lines)
        structure_chunks = joined_lines.split(self.TRIPOS_FLAG + self.MOLECULE_FLAG)
        entry_num = 0 if len(structure_chunks) > 2 else None
        for struct_chunk in structure_chunks:
            if struct_chunk != "":
                self._structures.append(self.parse_structure(struct_chunk, chunk_index=entry_num))

        if len(structure_chunks) - 1 != len(self._structures):
            logger.log(
                logging.WARN,
                "Only "
                + str(len(self._structures))
                + " structures could be parsed from "
                + str(len(structure_chunks) - 1)
                + " MOLECULE entries in the .mol2 file",
            )

    def parse_atoms(self, atom_lines: List[str]) -> List[Atom]:
        """Returns the Atom objects parsed from the atom_lines given.

        Args:
            atom_lines (List[str]): lines from the mol2 file pertaining to the atoms in the structure.

        Returns:
            List[Atom]: Atom objects parsed from atom_lines
        """
        atoms = []
        for atom_entry in atom_lines:
            if atom_entry == "" or atom_entry.strip() == self.ATOM_FLAG:
                continue
            atom_split = atom_entry.split()
            atoms.append(
                Atom(
                    index=int(atom_split[0]),
                    element=atom_split[1],
                    coords=atom_split[2:5],
                    atom_type_name=atom_split[5],
                    partial_charge=atom_split[8],
                )
            )
        return atoms

    def parse_bonds(self, bond_lines: List[str], structure: Structure) -> List[Bond]:
        """Returns the Bond objects parsed from the bond_lines given.

        Args:
            bond_lines (List[str]): lines from the mol2 file pertaining to the bond connectivity in the structure.
            structure (Structure): structure which the bonds pertain to, used for bond measurement.

        Returns:
            List[Bond]: Bond objects parsed from bond_lines
        """
        bonds = []
        for bond_entry in bond_lines:
            if bond_entry == "" or bond_entry.strip() == self.BOND_FLAG:
                continue
            bond_split = bond_entry.split()
            a_index = int(bond_split[1])
            b_index = int(bond_split[2])
            bonds.append(
                Bond(
                    atom_nums=[a_index, b_index],
                    order=int(bond_split[3]),
                    value=utilities.measure_bond(
                        structure.atoms[a_index - 1].coords,
                        structure.atoms[b_index - 1].coords,
                    ),
                )
            )

        # TODO: Ideally, the bonds class would measure the bonds and just contain a pointer to an Atom
        # object, but that would require a decent-sized refactor so hold off for now

        return bonds

    def parse_structure(self, structure_chunk: str, chunk_index:int = None) -> Structure:
        """Returns the Structure objects parsed from the structure_chunk given.

        Args:
            structure_chunk (str): string containing the lines which pertain to a single structure.

        Returns:
            Structure: the Structure object parsed from structure_chunk data.
        """
        tripos_chunks = structure_chunk.split(self.TRIPOS_FLAG)
        molecule_lines = tripos_chunks[0].split("\n")
        atom_lines = tripos_chunks[1].split("\n")
        bond_chunk = 2
        bond_lines = tripos_chunks[bond_chunk].split("\n")

        # assert that data was chunked correctly:
        assert atom_lines[0].strip() == self.ATOM_FLAG
        while bond_lines[0].strip() != self.BOND_FLAG:
            bond_chunk += 1
            try:
                bond_lines = tripos_chunks[bond_chunk].split("\n")
            except IndexError:
                logger.log(
                    logging.ERROR,
                    "No BOND flag within mol2 MOLECULE, invalid structure.",
                )

        # parse number of atoms and number of bonds from line 2 below @<TRIPOS>MOLECULE
        molecule_data = molecule_lines[2].split()
        num_atoms = int(molecule_data[0])
        num_bonds = int(molecule_data[1])

        file_identifier = self.filename if chunk_index is None else self.filename + str(chunk_index)

        struct = (
            Structure(file_identifier)
        )  # ideally we would gather data, then instantiate a Structure with
        # all the data as arguments, but for now I will follow the precedent within the Q2MM code
        # to avoid significant refactoring since we still don't have test cases or test scripts

        # send chunk from @<TRIPOS>ATOM to @<TRIPOS>BOND to parse_atoms
        struct._atoms = self.parse_atoms(atom_lines)

        # use num atoms from @<TRIPOS>MOLECULE to verify parse is correct
        assert (
            len(struct._atoms) == num_atoms
        ), "Parsed {} atoms but only expected {} atoms based on Mol2 data.".format(
            len(struct.atoms), num_atoms
        )
        assert all(
            struct.atoms[i].index == i + 1 for i in range(len(struct.atoms))
        ), "Mol2 atom index values do not match their ordering."

        # send chunk from @<TRIPOS>BOND to end-of-file to parse_bonds
        struct._bonds = self.parse_bonds(bond_lines, struct)

        # use num bonds from @<TRIPOS>MOLECULE to verify parse is correct
        assert (
            len(struct._bonds) == num_bonds
        ), "Parsed {} bonds but only expected {} bonds based on Mol2 data.".format(
            len(struct.bonds), num_bonds
        )

        return struct

    def value_bonds(
        self,
    ):  # TODO Not currently in use, remove if not needed by March 1, 2024.
        atom_list = self._structures.atoms
        for bond in self._structures.bonds:
            # Indexing atom_list is possible only because this is within the Mol2 class so
            # we can assume that the atoms were added in order of their atom index.
            atom1 = atom_list[bond.atom_nums[0] - 1]
            atom2 = atom_list[bond.atom_nums[1] - 1]
            bond.value = utilities.measure_bond(
                np.array(atom1.coords), np.array(atom2.coords)
            )


class GaussLog(File):
    """
    Retrieves data from Gaussian log files.

    If you are extracting frequencies/Hessian data from this file, use
    the keyword NoSymmetry when running the Gaussian calculation.
    """

    __slots__ = [
        "_lines",
        "path",
        "directory",
        "filename",
        "_evals",
        "_evecs",
        "_structures",
        "_esp_rms",
        "_au_hessian",
    ]

    def __init__(self, path: str, au_hessian=False):
        """Instantiates a file object for the file at the location path passed.

        Populates the directory and filename properties as well.

        Args:
            path (str): location of the Gaussian log file
            au_hessian (bool, optional): If true, Hessian will not be converted to
            kJ/(mol*Angstrom^2) but rather left in Atomic Units (AU) (Hartree/Bohr^2).
            Defaults to False.
        """
        super(GaussLog, self).__init__(path)
        self._evals = None
        self._evecs = None
        self._structures = None
        self._esp_rms = None
        self._au_hessian = au_hessian

    @property
    def evecs(self):
        """Returns eigenvectors of frequency analysis if applicable.  If not yet parsed,
        parses them from the log body, not the archive.

        Returns:
            TODO : eigenvectors of Gaussian frequency analysis
        """
        if self._evecs is None:
            self.read_out()
        return self._evecs

    @property
    def evals(self):
        """Returns eigenvalues of frequency analysis if applicable.  If not yet parsed,
        parses them from the log body, not the archive.

        Returns:
            TODO : eigenvalues of Gaussian frequency analysis
        """
        if self._evals is None:
            self.read_out()
        return self._evals

    @property
    def structures(self) -> List[Structure]:
        """Returns Structure objects parsed from the Gaussian log file. If None,
        parses the archive of the log file for structures.

        Returns:
            List[Structure]: Structures parsed from log file archive.
        """
        if self._structures is None:
            # self.read_out()
            self.read_archive()
        return self._structures

    @property
    def esp_rms(self):
        """Returns the esp_rms (Electrostatic potential ?? TODO)

        Returns:
            int | float: TODO
        """
        if self._esp_rms is None:
            self._esp_rms = -1
            self.read_out()
        return self._esp_rms

    def read_out(self):
        """
        Read force constant and eigenvector data from a frequency
        calculation.
        """
        logger.log(5, "READING: {}".format(self.filename))
        self._evals = []
        self._evecs = []
        self._structures = []
        force_constants = []
        evecs = []
        with open(self.path, "r") as f:
            # The keyword "harmonic" shows up before the section we're
            # interested in. It can show up multiple times depending on the
            # options in the Gaussian .com file.
            past_first_harm = False
            # High precision mode, turned on by including "freq=hpmodes" in the
            # Gaussian .com file.
            hpmodes = False
            file_iterator = iter(f)
            # This while loop breaks when the end of the file is reached, or
            # if the high quality modes have been read already.
            while True:
                try:
                    line = next(file_iterator)
                except:
                    # End of file.
                    break
                if "Charges from ESP fit" in line:
                    pattern = re.compile("RMS=\s+({0})".format(co.RE_FLOAT))
                    match = pattern.search(line)
                    self._esp_rms = float(match.group(1))
                # Gathering some geometric information.
                elif "Standard orientation:" in line:
                    self._structures.append(Structure(self.filename))
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    line = next(file_iterator)
                    while not "---" in line:
                        cols = line.split()
                        self._structures[-1].atoms.append(
                            Atom(
                                index=int(cols[0]),
                                atomic_num=int(cols[1]),
                                x=float(cols[3]),
                                y=float(cols[4]),
                                z=float(cols[5]),
                            )
                        )
                        line = next(file_iterator)
                    logger.log(
                        5,
                        "  -- Found {} atoms.".format(len(self._structures[-1].atoms)),
                    )
                elif "Harmonic" in line:
                    # The high quality eigenvectors come before the low quality
                    # ones. If you see "Harmonic" again, it means you're at the
                    # low quality ones now, so break.
                    if past_first_harm:
                        break
                    else:
                        past_first_harm = True
                elif "Frequencies" in line:
                    # We're going to keep reusing these.
                    # We accumulate sets of eigevectors and eigenvalues, add
                    # them to self._evecs and self._evals, and then reuse this
                    # for the next set.
                    del force_constants[:]
                    del evecs[:]
                    # Values inside line look like:
                    #     "Frequencies --- xxxx.xxxx xxxx.xxxx"
                    # That's why we remove the 1st two columns. This is
                    # consistent with and without "hpmodes".
                    # For "hpmodes" option, there are 5 of these frequencies.
                    # Without "hpmodes", there are 3.
                    # Thus the eigenvectors and eigenvalues will come in sets of
                    # either 5 or 3.
                    cols = line.split()
                    for frequency in map(float, cols[2:]):
                        # Has 1. or -1. depending on the sign of the frequency.
                        if frequency < 0.0:
                            force_constants.append(-1.0)
                        else:
                            force_constants.append(1.0)
                        # For now this is empty, but we will add to it soon.
                        evecs.append([])

                    # Moving on to the reduced masses.
                    line = next(file_iterator)
                    cols = line.split()
                    # Again, trim the "Reduced masses ---".
                    # It's "Red. masses --" for without "hpmodes".
                    for i, mass in enumerate(map(float, cols[3:])):
                        # +/- 1 / reduced mass
                        force_constants[i] = force_constants[i] / mass

                    # Now we are on the line with the force constants.
                    line = next(file_iterator)
                    cols = line.split()
                    # Trim "Force constants ---". It's "Frc consts --" without
                    # "hpmodes".
                    for i, force_constant in enumerate(map(float, cols[3:])):
                        # co.AU_TO_MDYNA = 15.569141
                        force_constants[i] *= force_constant / co.AU_TO_MDYNA

                    # Force constants were calculated above as follows:
                    #    a = +/- 1 depending on the sign of the frequency
                    #    b = a / reduced mass (obtained from the Gaussian log)
                    #    c = b * force constant / conversion factor (force
                    #         (constant obtained from Gaussian log) (conversion
                    #         factor is inside constants module)

                    # Skip the IR intensities.
                    next(file_iterator)
                    # This is different depending on whether you use "hpmodes".
                    line = next(file_iterator)
                    # "Coord" seems to only appear when the "hpmodes" is used.
                    if "Coord" in line:
                        hpmodes = True
                    # This is different depending on whether you use
                    # "freq=projected".
                    line = next(file_iterator)
                    # The "projected" keyword seems to add "IRC Coupling".
                    if "IRC Coupling" in line:
                        line = next(file_iterator)
                    # We're on to the eigenvectors.
                    # Until the end of this section containing the eigenvectors,
                    # the number of columns remains constant. When that changes,
                    # we know we're to the next set of frequencies, force
                    # constants and eigenvectors.
                    # Actually check that we've moved on, sometimes a "Depolar" entry is
                    if "Depolar" in line:
                        line = next(file_iterator)
                    if "Atom" in line:
                        line = next(file_iterator)
                    cols = line.split()
                    cols_len = len(cols)

                    while len(cols) == cols_len:
                        # This will come after all the eigenvectors have been
                        # read. We can break out then.
                        if "Harmonic" in line:
                            break
                        # If "hpmodes" is used, you have an extra column here
                        # that is simply an index.
                        if hpmodes:
                            cols = cols[1:]
                        # cols corresponds to line(s) (maybe only 1st line)
                        # under section "Coord Atom Element:" (at least for
                        # "hpmodes").

                        # Just the square root of the mass from co.MASSES.
                        # co.MASSES currently has the average mass.
                        # Gaussian may use the mass of the most abundant
                        # isotope. This may be a problem.
                        mass_sqrt = np.sqrt(
                            list(co.MASSES.items())[int(cols[1]) - 1][1]
                        )

                        cols = cols[2:]
                        # This corresponds to the same line still, but without
                        # the atom elements.

                        # This loop expands the LoL, evecs, as so.
                        # Iteration 1:
                        # [[x], [x], [x], [x], [x]]
                        # Iteration 2:
                        # [[x, x], [x, x], [x, x], [x, x], [x, x]]
                        # ... etc. until the length of the sublist is equal to
                        # the number of atoms. Remember, for low precision
                        # eigenvectors it only adds in sets of 3, not 5.

                        # Elements of evecs are simply the data under
                        # "Coord Atom Element" multiplied by the square root
                        # of the weight.
                        for i in range(len(evecs)):
                            if hpmodes:
                                # evecs is a LoL. Length of sublist is
                                # equal to # of columns in section "Coord Atom
                                # Element" minus 3, for the 1st 3 columns
                                # (index, atom index, atomic number).
                                evecs[i].append(float(cols[i]) * mass_sqrt)
                            else:
                                # This is fow low precision eigenvectors. It's a
                                # funny way to go in sets of 3. Take a look at
                                # your low precision Gaussian log and it will
                                # make more sense.
                                for useless in range(3):
                                    x = float(cols.pop(0))
                                    evecs[i].append(x * mass_sqrt)
                        line = next(file_iterator)
                        cols = line.split()

                    # Here the overall number of eigenvalues and eigenvectors is
                    # increased by 5 (high precision) or 3 (low precision). The
                    # total number goes to 3N - 6 for non-linear and 3N - 5 for
                    # linear. Same goes for self._evecs.
                    for i in range(len(evecs)):
                        self._evals.append(force_constants[i])
                        self._evecs.append(evecs[i])
                    # We know we're done if this is in the line.
                    if "Harmonic" in line:
                        break
        if self._evals and self._evecs:
            for evec in self._evecs:
                # Each evec is a single eigenvector.
                # Add up the sum of squares over an eigenvector.
                sum_of_squares = 0.0
                # Appropriately named, element is an element of that single
                # eigenvector.
                for element in evec:
                    sum_of_squares += element * element
                # Now x is the inverse of the square root of the sum of squares
                # for an individual eigenvector.
                element = 1 / np.sqrt(sum_of_squares)
                for i in range(len(evec)):
                    evec[i] *= element
            self._evals = np.array(self._evals)
            self._evecs = np.array(self._evecs)
            logger.log(1, ">>> self._evals: {}".format(self._evals))
            logger.log(1, ">>> self._evecs: {}".format(self._evecs))
            logger.log(5, "  -- {} structures found.".format(len(self.structures)))

    # May want to move some attributes assigned to the structure class onto
    # this filetype class.
    def read_archive(self):
        """
        Only reads last archive found in the Gaussian .log file. Hessian converted
        to kJ/molA^2
        """
        logger.log(5, "READING: {}".format(self.filename))
        struct = Structure(self.filename)
        self._structures = [struct]
        # Matches everything in between the start and end.
        # (?s)  - Flag for re.compile which says that . matches all.
        # \\\\  - One single \
        # Start - " 1\1\".
        # End   - Some number of \ followed by @. Not sure how many \ there
        #         are, so this matches as many as possible. Also, this could
        #         get separated by a line break (which would also include
        #         adding in a space since that's how Gaussian starts new lines
        #         in the archive).
        # We pull out the last one [-1] in case there are multiple archives
        # in a file.
        #        print(self.path)
        #        print(open(self.path,'r').read())
        #        print(re.findall('(?s)(\s1\\\\1\\\\.*?[\\\\\n\s]+@)',open(self.path,'r').read()))
        try:
            arch = re.findall(
                "(?s)(\s1\\\\1\\\\.*?[\\\\\n\s]+@)", open(self.path, "r").read()
            )[-1]
            logger.log(5, "  -- Located last archive.")
        except IndexError:
            logger.warning("  -- Couldn't locate archive.")
            raise
        # Make it into one string.
        arch = arch.replace("\n ", "")
        # Separate it by Gaussian's section divider.
        arch = arch.split("\\\\")
        # Helps us iterate over sections of the archive.
        section_counter = 0
        # SECTION 0
        # General job information.
        arch_general = arch[section_counter]
        section_counter += 1
        stuff = re.search(
            "\s1\\\\1\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\(?P<user>.*?)"
            "\\\\(?P<date>.*?)"
            "\\\\.*?",
            arch_general,
        )
        struct.props["user"] = stuff.group("user")
        struct.props["date"] = stuff.group("date")
        # SECTION 1
        # The commands you wrote.
        arch_commands = arch[section_counter]
        section_counter += 1
        # SECTION 2
        # The comment line.
        arch_comment = arch[section_counter]
        section_counter += 1
        # SECTION 3
        # Actually has charge, multiplicity and coords.
        arch_coords = arch[section_counter]
        section_counter += 1
        stuff = re.search(
            "(?P<charge>.*?)" ",(?P<multiplicity>.*?)" "\\\\(?P<atoms>.*)", arch_coords
        )
        struct.props["charge"] = stuff.group("charge")
        struct.props["multiplicity"] = stuff.group("multiplicity")
        # We want to do more fancy stuff with the atoms than simply add to
        # the properties dictionary.
        atoms = stuff.group("atoms")
        atoms = atoms.split("\\")
        # Z-matrix coordinates adds another section. We need to be aware of
        # this.
        probably_z_matrix = False
        for atom in atoms:
            stuff = atom.split(",")
            # An atom typically looks like this:
            #    C,0.1135,0.13135,0.63463
            if len(stuff) == 4:
                ele, x, y, z = stuff
            # But sometimes they look like this (notice the extra zero):
            #    C,0,0.1135,0.13135,0.63463
            # I'm not sure what that extra zero is for. Anyway, ignore
            # that extra whatever if it's there.
            elif len(stuff) == 5:
                ele, x, y, z = stuff[0], stuff[2], stuff[3], stuff[4]
            # And this would be really bad. Haven't seen anything else like
            # this yet.
            # 160613 - So, not sure when I wrote that comment, but something
            # like this definitely happens when using scans and z-matrices.
            # I'm going to ignore grabbing any atoms in this case.
            else:
                logger.warning(
                    "Not sure how to read coordinates from Gaussian acrhive!"
                )
                probably_z_matrix = True
                section_counter += 1
                # Let's have it stop looping over atoms, but not fail anymore.
                break
                # raise Exception(
                #     'Not sure how to read coordinates from Gaussian archive!')
            struct._atoms.append(Atom(element=ele, x=float(x), y=float(y), z=float(z)))
        logger.log(20, "  -- Read {} atoms.".format(len(struct._atoms)))
        # SECTION 4
        # All sorts of information here. This area looks like:
        #     prop1=value1\prop2=value2\prop3=value3
        arch_info = arch[section_counter]
        section_counter += 1
        arch_info = arch_info.split("\\")
        for thing in arch_info:
            prop_name, prop_value = thing.split("=")
            struct.props[prop_name] = prop_value
        # SECTION 5
        # The Hessian. Only exists if you did a frequency calculation.
        # Appears in lower triangular form, not mass-weighted.
        if not arch[section_counter] == "@":
            hess_tri = arch[section_counter]
            hess_tri = hess_tri.split(",")
            logger.log(
                5,
                "  -- Read {} Hessian elements in lower triangular "
                "form.".format(len(hess_tri)),
            )
            hess = np.zeros([len(atoms) * 3, len(atoms) * 3], dtype=float)
            logger.log(5, "  -- Created {} Hessian matrix.".format(hess.shape))
            # Code for if it was in upper triangle (it's not).
            # hess[np.triu_indices_from(hess)] = hess_tri
            # hess += np.triu(hess, -1).T
            # Lower triangle code.
            hess[np.tril_indices_from(hess)] = hess_tri
            hess += np.tril(hess, -1).T
            if not self._au_hessian:
                hess *= co.HESSIAN_CONVERSION
            struct.hess = hess
            # SECTION 6
            # Not sure what this is.

        # stuff = re.search(
        #     '\s1\\\\1\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\(?P<user>.*?)'
        #     '\\\\(?P<date>.*?)'
        #     '\\\\.*?\\\\\\\\(?P<com>.*?)'
        #     '\\\\\\\\(?P<filename>.*?)'
        #     '\\\\\\\\(?P<charge>.*?)'
        #     ',(?P<multiplicity>.*?)'
        #     '\\\\(?P<atoms>.*?)'
        #     # This marks the end of what always shows up.
        #     '\\\\\\\\'
        #     # This stuff sometimes shows up.
        #     # And it breaks if it doesn't show up.
        #     '.*?HF=(?P<hf>.*?)'
        #     '\\\\.*?ZeroPoint=(?P<zp>.*?)'
        #     '\\\\.*?Thermal=(?P<thermal>.*?)'
        #     '\\\\.*?\\\\NImag=\d+\\\\\\\\(?P<hess>.*?)'
        #     '\\\\\\\\(?P<evals>.*?)'
        #     '\\\\\\\\\\\\',
        #     arch)
        # logger.log(5, '  -- Read archive.')
        # atoms = stuff.group('atoms')
        # atoms = atoms.split('\\')
        # for atom in atoms:
        #     ele, x, y, z = atom.split(',')
        #     struct.atoms.append(
        #         Atom(element=ele, x=float(x), y=float(y), z=float(z)))
        # logger.log(5, '  -- Read {} atoms.'.format(len(atoms)))
        # self._structures = [struct]
        # hess_tri = stuff.group('hess')
        # hess_tri = hess_tri.split(',')
        # logger.log(
        #     5,
        #     '  -- Read {} Hessian elements in lower triangular '
        #     'form.'.format(len(hess_tri)))
        # hess = np.zeros([len(atoms) * 3, len(atoms) * 3], dtype=float)
        # logger.log(
        #     5, '  -- Created {} Hessian matrix.'.format(hess.shape))
        # # Code for if it was in upper triangle, but it's not.
        # # hess[np.triu_indices_from(hess)] = hess_tri
        # # hess += np.triu(hess, -1).T
        # # Lower triangle code.
        # hess[np.tril_indices_from(hess)] = hess_tri
        # hess += np.tril(hess, -1).T
        # hess *= co.HESSIAN_CONVERSION
        # struct.hess = hess
        # # Code to extract energies.
        # # Still not sure exactly what energies we want to use.
        # struct.props['hf'] = float(stuff.group('hf'))
        # struct.props['zp'] = float(stuff.group('zp'))
        # struct.props['thermal'] = float(stuff.group('thermal'))

    def get_most_converged(self, structures=None):
        """
        Used with geometry optimizations that don't succeed. Sometimes
        intermediate geometries obtain better convergence than the
        final geometry. This function returns the class Structure for
        the most converged geometry, which can then be used to output
        the coordinates for the next optimization.
        """
        if structures is None:
            structures = self.structures
        structures_compared = 0
        best_structure = None
        best_yes_or_no = None
        fields = [
            "RMS Force",
            "RMS Displacement",
            "Maximum Force",
            "Maximum Displacement",
        ]
        for i, structure in reversed(list(enumerate(structures))):
            yes_or_no = [
                value[2] for key, value in structure.props.items() if key in fields
            ]
            if not structure._atoms:
                logger.warning(
                    "  -- No atoms found in structure {}. " "Skipping.".format(i + 1)
                )
                continue
            if len(yes_or_no) == 4:
                structures_compared += 1
                if best_structure is None:
                    logger.log(10, "  -- Most converged structure: {}".format(i + 1))
                    best_structure = structure
                    best_yes_or_no = yes_or_no
                elif yes_or_no.count("YES") > best_yes_or_no.count("YES"):
                    best_structure = structure
                    best_yes_or_no = yes_or_no
                elif yes_or_no.count("YES") == best_yes_or_no.count("YES"):
                    number_better = 0
                    for field in fields:
                        if structure.props[field][0] < best_structure.props[field][0]:
                            number_better += 1
                    if number_better > 2:
                        best_structure = structure
                        best_yes_or_no = yes_or_no
            elif len(yes_or_no) != 0:
                logger.warning(
                    "  -- Partial convergence criterion in structure: {}".format(
                        self.path
                    )
                )
        logger.log(
            10,
            "  -- Compared {} out of {} structures.".format(
                structures_compared, len(self.structures)
            ),
        )
        return best_structure

    def read_optimization(self, coords_type="both"):
        """
        Finds structures from a Gaussian geometry optimization that
        are listed throughout the log file. Also finds data about
        their convergence.

        coords_type = "input" or "standard" or "both"
                      Using both may cause coordinates in one format
                      to be overwritten by whatever comes later in the
                      log file.
        """
        logger.log(10, "READING: {}".format(self.filename))
        structures = []
        with open(self.path, "r") as f:
            section_coords_input = False
            section_coords_standard = False
            section_convergence = False
            section_optimization = False
            for i, line in enumerate(f):
                # Look for start of optimization section of log file and
                # set a flag that it has indeed started.
                if section_optimization and "Optimization stopped." in line:
                    section_optimization = False
                    logger.log(5, "[L{}] End optimization section.".format(i + 1))
                if not section_optimization and "Search for a local minimum." in line:
                    section_optimization = True
                    logger.log(5, "[L{}] Start optimization section.".format(i + 1))
                if section_optimization:
                    # Start of a structure.
                    if "Step number" in line:
                        structures.append(Structure(self.filename))
                        current_structure = structures[-1]
                        logger.log(
                            5,
                            "[L{}] Added structure "
                            "(currently {}).".format(i + 1, len(structures)),
                        )
                    # Look for convergence information related to a single
                    # structure.
                    if section_convergence and "GradGradGrad" in line:
                        section_convergence = False
                        logger.log(5, "[L{}] End convergence section.".format(i + 1))
                    if section_convergence:
                        match = re.match(
                            "\s(Maximum|RMS)\s+(Force|Displacement)\s+({0})\s+"
                            "({0})\s+(YES|NO)".format(co.RE_FLOAT),
                            line,
                        )
                        if match:
                            current_structure.props[
                                "{} {}".format(match.group(1), match.group(2))
                            ] = (
                                float(match.group(3)),
                                float(match.group(4)),
                                match.group(5),
                            )
                    if "Converged?" in line:
                        section_convergence = True
                        logger.log(5, "[L{}] Start convergence section.".format(i + 1))
                    # Look for input coords.
                    if coords_type == "input" or coords_type == "both":
                        # End of input coords for a given structure.
                        if section_coords_input and "Distance matrix" in line:
                            section_coords_input = False
                            logger.log(
                                5,
                                "[L{}] End input coordinates section "
                                "({} atoms).".format(i + 1, count_atom),
                            )
                        # Add atoms and coords to structure.
                        if section_coords_input:
                            match = re.match(
                                "\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+({0})\s+"
                                "({0})".format(co.RE_FLOAT),
                                line,
                            )
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1)) - 1
                                    ]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num:
                                    assert current_atom.atomic_num == int(
                                        match.group(2)
                                    ), (
                                        "[L{}] Atomic numbers don't match "
                                        "(current != existing) "
                                        "({} != {}).".format(
                                            i + 1,
                                            int(match.group(2)),
                                            current_atom.atomic_num,
                                        )
                                    )
                                else:
                                    current_atom.atomic_num = int(match.group(2))
                                current_atom.index = int(match.group(1))
                                current_atom.coords_type = "input"
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # Start of input coords for a given structure.
                        if not section_coords_input and "Input orientation:" in line:
                            section_coords_input = True
                            count_atom = 0
                            logger.log(
                                5,
                                "[L{}] Start input coordinates "
                                "section.".format(i + 1),
                            )
                    # Look for standard coords.
                    if coords_type == "standard" or coords_type == "both":
                        # End of coordinates for a given structure.
                        if section_coords_standard and (
                            "Rotational constants" in line or "Leave Link" in line
                        ):
                            section_coords_standard = False
                            logger.log(
                                5,
                                "[L{}] End standard coordinates "
                                "section ({} atoms).".format(i + 1, count_atom),
                            )
                        # Grab coords for each atom. Add atoms to the structure.
                        if section_coords_standard:
                            match = re.match(
                                "\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+"
                                "({0})\s+({0})".format(co.RE_FLOAT),
                                line,
                            )
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1)) - 1
                                    ]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num:
                                    assert current_atom.atomic_num == int(
                                        match.group(2)
                                    ), (
                                        "[L{}] Atomic numbers don't match "
                                        "(current != existing) "
                                        "({} != {}).".format(
                                            i + 1,
                                            int(match.group(2)),
                                            current_atom.atomic_num,
                                        )
                                    )
                                else:
                                    current_atom.atomic_num = int(match.group(2))
                                current_atom.index = int(match.group(1))
                                current_atom.coords_type = "standard"
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # Start of standard coords.
                        if (
                            not section_coords_standard
                            and "Standard orientation" in line
                        ):
                            section_coords_standard = True
                            count_atom = 0
                            logger.log(
                                5,
                                "[L{}] Start standard coordinates "
                                "section.".format(i + 1),
                            )
        return structures


# Row of mm3.fld where comments start.
COM_POS_START = 96
# Row where standard 3 columns of parameters appear.
P_1_START = 23
P_1_END = 33
P_2_START = 34
P_2_END = 44
P_3_START = 45
P_3_END = 55


class ParamError(Exception):
    pass


class ParamFE(Exception):
    pass


class ParamBE(Exception):
    pass


class Param(object):
    """
     A single parameter of a force field (FF). TODO rework this to match Google style docstrings
     for later sphinx autodocumentation.

     :var _allowed_range: Stored as None if not set, else it's set to True or
       False depending on :func:`allowed_range`.
    :type _allowed_range: None, 'both', 'pos', 'neg'

     :ivar ptype: Parameter type can be one of the following: ae, af, be, bf, df,
       imp1, imp2, sb, or q.
     :type ptype: string

     Attributes
     ----------
     d1 : float
          First derivative of parameter with respect to penalty function.
     d2 : float
          Second derivative of parameter with respect to penalty function.
     step : float
            Step size used during numerical differentiation.
     ptype : {'ae', 'af', 'be', 'bf', 'df', 'imp1', 'imp2', 'sb', 'q'}
     value : float
             Value of the parameter.
    """

    __slots__ = ["_allowed_range", "_step", "_value", "d1", "d2", "ptype", "simp_var"]

    def __init__(
        self, d1: float = None, d2: float = None, ptype=None, value: float = None
    ):
        """_summary_

        Args:
            d1 (float, optional): First derivative of parameter with respect to penalty function. Defaults to None.
            d2 (float, optional): Second derivative of parameter with respect to penalty function. Defaults to None.
            ptype (_type_, optional): Parameter type {'ae', 'af', 'be', 'bf', 'df', 'imp1', 'imp2', 'sb', 'q'}. Defaults to None.
            value (float, optional): Value of the parameter. Defaults to None.
        """
        self._allowed_range = None
        self._step = None
        self._value = None
        self.d1 = d1
        self.d2 = d2
        self.ptype = ptype
        self.simp_var = None
        self.value = value

    def __repr__(self):
        return "{}[{}]({:7.4f})".format(self.__class__.__name__, self.ptype, self.value)

    @property
    def allowed_range(self) -> List[float]:
        """Returns the allowed range of values for the parameter based on its parameter type (ptype).

        Returns:
            List[float]: [minimum_value, maximum_value]
        """
        if self._allowed_range is None and self.ptype is not None:
            if self.ptype in ["q", "df"]:
                self._allowed_range = [-float("inf"), float("inf")]
            else:
                self._allowed_range = [0.0, float("inf")]
        return self._allowed_range

    @property
    def step(self):
        """TODO Google style
        Returns a float for the current step size that should be used. If
        _step is a string, return float(_step) * value. If
        _step is a float, simply return that.

        Not sure how well the check for a step size of zero works.
        """
        if self._step is None:
            try:
                self._step = co.STEPS[self.ptype]
            except KeyError:
                logger.warning(
                    "{} doesn't have a default step size and none "
                    "provided!".format(self)
                )
                raise
        if sys.version_info > (3, 0):
            if isinstance(self._step, str):
                return float(self._step) * self.value
            else:
                return self._step
        else:
            if isinstance(self._step, basestring):
                return float(self._step) * self.value
            else:
                return self._step

    @step.setter
    def step(self, x):
        self._step = x

    @property
    def value(self):
        if self.ptype == "ae" and self._value > 180.0:
            self._value = 180.0 - abs(180 - self._value)
        return self._value

    @value.setter
    def value(self, value):
        """TODO Google style
        When you try to give the parameter a value, make sure that's okay.
        """
        if self.value_in_range(value):
            self._value = value

    def convert_and_set(self, value: float, units=None):
        """Converts force constant value in kJ/molA to the correct units based on FF units and parameter type.

        Note: This should only be used for force constants, not equilibrium bond lengths or angles or charges.

        Args:
            value (float): New value for the parameter
            units (str, optional): units to convert to for FF, must be in constants.py. Defaults to None.
        """
        if value is None:
            return
        if units == co.MM3FF:
            self.value = value / co.MM3_STR #  Uses the conversion factor specific to MM3.fld, Notes on this in box TODO: Remove in a later commit and note commit # in documentation
            # self.value = value / (co.HARTREE_TO_KJMOL * co.BOHR_TO_ANG**2)  if self.ptype == 'bf' else value / (co.HARTREE_TO_KJMOL * co.BOHR_TO_ANG)
            # self.value = value * co.AU_TO_MDYNA  if self.ptype == 'bf' else value * co.AU_TO_MDYN_ANGLE
            # self.value = value * 10**6  if self.ptype == 'bf' else value * co.KJMOLA_TO_MDYN
            # self.value = (
            #     value / co.MDYNA_TO_KJMOLA2
            #     if self.ptype == "bf"
            #     else value * co.KJMOLA_TO_MDYN
            # )
        elif units == co.AMBERFF:  # TODO Amber conversion factor is unknown, ask David Case because it is not just units.
            self.value = (
                value * co.HARTREE_TO_KCALMOL / (co.BOHR_TO_ANG**2)
                if self.ptype == "bf"
                else value * co.HARTREE_TO_KCALMOL
            )
        elif units == co.TINKERFF:
            raise NotImplemented()
        else:
            raise Exception(
                "Only MM3, AMBER, and Tinker type force fields have defined units and conversions for parameters in Q2MM."
            )

    def value_in_range(self, value):
        """TODO

        Args:
            value (_type_): _description_

        Raises:
            ParamBE: _description_
            ParamFE: _description_
            ParamError: _description_

        Returns:
            _type_: _description_
        """
        if self.allowed_range[0] <= value <= self.allowed_range[1]:
            return True
        elif value == self.allowed_range[0] - 0.1:
            raise ParamBE(
                "{} Backward Error. Forward Derivative only".format(str(self))
            )
        elif value == self.allowed_range[1] + 0.1:
            raise ParamFE(
                "{} Forward Error. Backward Derivative only".format(str(self))
            )
        elif value == self.allowed_range[1] or value == self.allowed_range[0]:
            return True
        else:
            raise ParamError(
                "{} isn't allowed to have a value of {}! "
                "({} <= x <= {})".format(
                    str(self), value, self.allowed_range[0], self.allowed_range[1]
                )
            )

    def value_at_limits(self):
        """TODO"""
        # Checks if the parameter is at the limits of
        # its allowed range. Should only be run at the
        # end of an optimization to warn users they should
        # consider whether this is ok.
        if self.value == min(self.allowed_range):
            logger.warning(
                "{} is equal to its lower limit of {}!\nReconsider "
                "if you need to adjust limits, initial parameter "
                "values, or if your reference data is appropriate.".format(
                    str(self), self.value
                )
            )
        if self.value == max(self.allowed_range):
            logger.warning(
                "{} is equal to its upper limit of {}!\nReconsider "
                "if you need to adjust limits, initial parameter "
                "values, or if your reference data is appropriate.".format(
                    str(self), self.value
                )
            )


# TODO: MF - I see little reason to have ParamMM3 or ParAMBER, having a singular Param object
# should suffice by using a Param.units or Param.ff_type variable. Perhaps consider this in future
# refactoring efforts.


# Need a general index scheme/method/property to compare the equalness of two
# parameters, rather than having to rely on some expression that compares
# mm3_row and mm3_col.
# MF - I agree, a __equal__ would be nice, but its use would require a refactor so I recommend for future.
class ParamMM3(Param):
    """
    Adds information to Param that is specific to MM3* parameters. TODO
    """

    __slots__ = ["atom_labels", "atom_types", "mm3_col", "mm3_row", "mm3_label"]

    def __init__(
        self,
        atom_labels=None,
        atom_types=None,
        mm3_col=None,
        mm3_row=None,
        mm3_label=None,
        d1=None,
        d2=None,
        ptype=None,
        value=None,
    ):
        self.atom_labels = atom_labels
        self.atom_types = atom_types
        self.mm3_col = mm3_col
        self.mm3_row = mm3_row
        self.mm3_label = mm3_label
        super(ParamMM3, self).__init__(ptype=ptype, value=value)

    def __repr__(self):
        return "{}[{}][{},{}]({})".format(
            self.__class__.__name__, self.ptype, self.mm3_row, self.mm3_col, self.value
        )

    def __str__(self):
        return "{}[{}][{},{}]({})".format(
            self.__class__.__name__, self.ptype, self.mm3_row, self.mm3_col, self.value
        )

    def convert_and_set(self, value):
        return super().convert_and_set(value, units=co.MM3FF)


class ParAMBER(Param):
    """
    Adds information to Param that is specific to AMBER parameters. TODO
    """

    __slots__ = ["atom_labels", "atom_types", "mm3_col", "mm3_row", "mm3_label"]

    def __init__(
        self,
        atom_labels=None,
        atom_types=None,
        mm3_col=None,
        mm3_row=None,
        mm3_label=None,
        d1=None,
        d2=None,
        ptype=None,
        value=None,
    ):
        self.atom_labels = atom_labels
        self.atom_types = atom_types
        self.mm3_col = mm3_col
        self.mm3_row = mm3_row
        self.mm3_label = mm3_label
        super(ParAMBER, self).__init__(ptype=ptype, value=value)

    def __repr__(self):
        return "{}[{}][{},{}]({})".format(
            self.__class__.__name__, self.ptype, self.mm3_row, self.mm3_col, self.value
        )

    def __str__(self):
        return "{}[{}][{},{}]({})".format(
            self.__class__.__name__, self.ptype, self.mm3_row, self.mm3_col, self.value
        )

    def convert_and_set(self, value):
        return super().convert_and_set(value, units=co.AMBERFF)


class Datum(object):
    """
    Class for a reference or calculated data point. TODO
    """

    __slots__ = [
        "_lbl",
        "val",
        "wht",
        "typ",
        "com",
        "src_1",
        "src_2",
        "idx_1",
        "idx_2",
        "atm_1",
        "atm_2",
        "atm_3",
        "atm_4",
        "ff_row",
    ]

    def __init__(
        self,
        lbl=None,
        val=None,
        wht=None,
        typ=None,
        com=None,
        src_1=None,
        src_2=None,
        idx_1=None,
        idx_2=None,
        atm_1=None,
        atm_2=None,
        atm_3=None,
        atm_4=None,
        ff_row=None,
    ):
        self._lbl = lbl
        self.val = val
        self.wht = wht
        self.typ = typ
        self.com = com
        self.src_1 = src_1
        self.src_2 = src_2
        self.idx_1 = idx_1
        self.idx_2 = idx_2
        self.atm_1 = atm_1
        self.atm_2 = atm_2
        self.atm_3 = atm_3
        self.atm_4 = atm_4
        self.ff_row = ff_row

    def __repr__(self):
        return "{}({:7.4f})".format(self.lbl, self.val)

    @property
    def lbl(self):
        if self._lbl is None:
            a = self.typ
            if self.src_1:
                b = re.split("[.]+", self.src_1)[0]
            # Why would it ever not have src_1?
            else:
                b = None
            c = "-".join([str(x) for x in remove_none(self.idx_1, self.idx_2)])
            d = "-".join(
                [
                    str(x)
                    for x in remove_none(self.atm_1, self.atm_2, self.atm_3, self.atm_4)
                ]
            )
            abcd = remove_none(a, b, c, d)
            self._lbl = "_".join(abcd)
        return self._lbl


def remove_none(*args):
    return [x for x in args if (x is not None and x != "")]


def datum_sort_key(datum):
    """
    Used as the key to sort a list of Datum instances. This should always ensure
    that the calculated and reference data points align properly.
    """
    return (datum.typ, datum.src_1, datum.src_2, datum.idx_1, datum.idx_2)


class FF(object):
    """TODO
    Class for any type of force field.

    path   - Self explanatory.
    data   - List of Datum objects.
    method - String describing method used to generate this FF.
    params - List of Param objects.
    score  - Float which is the objective function score.
    """

    __slots__ = ["path", "data", "method", "params", "score"]

    def __init__(self, path=None, data=None, method=None, params:list[Param]=None, score=None):
        self.path = path
        self.data = data
        self.method = method
        self.params: List[Param] = params
        self.score = score

    def copy_attributes(self, ff):
        """
        Copies some general attributes to another force field.

        Parameters
        ----------
        ff : `datatypes.FF`
        """
        ff.path = self.path

    def __repr__(self):
        return "{}[{}]({})".format(self.__class__.__name__, self.method, self.score)


class AmberFF(FF):
    """
    STUFF TO FILL IN LATER TODO
    """

    units = co.AMBERFF

    def __init__(self, path=None, data=None, method=None, params=None, score=None):
        super(AmberFF, self).__init__(path, data, method, params, score)
        self.sub_names = []
        self._atom_types = None
        self._lines = None
        # change constant
        co.STEPS["bf"] = 10.00
        co.STEPS["af"] = 10.0
        co.STEPS["df"] = 10.0

    def copy_attributes(self, ff):
        """
        Copies some general attributes to another force field.

        Parameters
        ----------
        """
        ff.path = self.path
        ff.sub_names = self.sub_names
        ff._atom_types = self._atom_types
        ff._lines = self._lines

    @property
    def lines(self):
        if self._lines is None:
            with open(self.path, "r") as f:
                self._lines = f.readlines()
        return self._lines

    @lines.setter
    def lines(self, x):
        self._lines = x

    def import_ff(self, path=None, sub_search="OPT"):
        if path is None:
            path = self.path
        bonds = ["bond", "bond3", "bond4", "bond5"]
        pibonds = ["pibond", "pibond3", "pibond4", "pibond5"]
        angles = ["angle", "angle3", "angle4", "angle5"]
        torsions = ["torsion", "torsion4", "torsion5"]
        dipoles = ["dipole", "dipole3", "dipole4", "dipole5"]
        self.params: List[Param] = []
        q2mm_sec = False
        gather_data = False
        self.sub_names = []
        count = 0
        with open(path, "r") as f:
            logger.log(15, "READING: {}".format(path))
            for i, line in enumerate(f):
                split = line.split()
                if not q2mm_sec and "# Q2MM" in line:
                    q2mm_sec = True
                elif q2mm_sec and "#" in line[0]:
                    self.sub_names.append(line[1:])
                    if "OPT" in line:
                        gather_data = True
                    else:
                        gather_data = False
                if gather_data and split:
                    if "MASS" in line and count == 0:
                        count = 1
                        continue
                    if "BOND" in line and count == 1:
                        count = 2
                        continue
                    elif count == 1 and "ANGL" not in line:
                        # atom symbol:atomic mass:atomic polarizability
                        at = split[0]  # need number if it matters
                        el = split[0]
                        mass = split[1]
                        if len(split) > 2:
                            pol = split[2]
                        # no need for atom label
                        # at = ["Z0", "P1", "CX"]
                    # BOND
                    if "ANGL" in line and count == 2:
                        count = 3
                        continue
                    elif count == 2 and "DIHE" not in line:
                        # A1-A2 Force Const in kcal/mol/(A**2): Eq. length in A
                        AA = line[:5].split("-")
                        BB = line[5:].split()
                        at = [AA[0], AA[1]]
                        self.params.extend(
                            (
                                ParAMBER(
                                    atom_types=at,
                                    ptype="bf",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    value=float(BB[0]),
                                ),
                                ParAMBER(
                                    atom_types=at,
                                    ptype="be",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    value=float(BB[1]),
                                ),
                            )
                        )
                    # ANGLE
                    if "DIHE" in line and count == 3:
                        count = 4
                        continue
                    elif count == 3 and "IMPR" not in line:
                        AA = line[: 2 + 3 * 2].split("-")
                        BB = line[2 + 3 * 2 :].split()
                        at = [AA[0], AA[1], AA[2]]
                        self.params.extend(
                            (
                                ParAMBER(
                                    atom_types=at,
                                    ptype="af",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    value=float(BB[0]),
                                ),
                                ParAMBER(
                                    atom_types=at,
                                    ptype="ae",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    value=float(BB[1]),
                                ),
                            )
                        )
                    # Dihedral
                    if "IMPR" in line and count == 4:
                        count = 5
                        continue
                    elif count == 4 and "NONB" not in line:
                        # (PK/IDIVF) * (1 + cos(PN*phi - PHASE))
                        # A4 IDIVF PK PHASE PN
                        nl = 2 + 3 * 3
                        AA = line[:nl].split("-")
                        BB = line[nl:].split()
                        at = [AA[0], AA[1], AA[2], AA[3]]
                        self.params.append(
                            ParAMBER(
                                atom_types=at,
                                ptype="df",
                                mm3_col=1,
                                mm3_row=i + 1,
                                value=float(BB[1]),
                            )
                        )

                    # Improper
                    if "NONB" in line and count == 5:
                        count = 6
                        continue
                    elif count == 5:
                        nl = 2 + 3 * 3
                        AA = line[:nl].split("-")
                        BB = line[nl:].split()
                        at = [AA[0], AA[1], AA[2], AA[3]]
                        self.params.append(
                            ParAMBER(
                                atom_types=at,
                                ptype="imp1",
                                mm3_col=1,
                                mm3_row=i + 1,
                                value=float(BB[0]),
                            )
                        )

                    #                    # Hbond
                    #                    if "NONB" in line and count == 6:
                    #                        count == 7
                    #                        continue
                    #                    elif count == 6:
                    #                        0

                    # NONB
                    if count == 6:
                        continue

                    if "vdw" == split[0]:
                        # The first float is the vdw radius, the second has to do
                        # with homoatomic well depths and the last is a reduction
                        # factor for univalent atoms (I don't think we will need
                        # any of these except for the first one).
                        at = [split[1]]
                        self.params.append(
                            ParAMBER(
                                atom_types=at,
                                ptype="vdw",
                                mm3_col=1,
                                mm3_row=i + 1,
                                value=float(split[2]),
                            )
                        )
        logger.log(15, "  -- Read {} parameters.".format(len(self.params)))

    def export_ff(self, path=None, params:List[Param]=None, lines=None):
        """
        Exports the force field to a file, typically mm3.fld.
        """
        if path is None:
            path = self.path
        if params is None:
            params:List[Param] = self.params
        if lines is None:
            lines = self.lines
        for param in params:
            logger.log(1, ">>> param: {} param.value: {}".format(param, param.value))
            line = lines[param.mm3_row - 1]
            if abs(param.value) > 1999.0:
                logger.warning("Value of {} is too high! Skipping write.".format(param))
            else:
                atoms = ""
                const = ""
                space3 = " " * 3
                col = int(param.mm3_col - 1)
                value = "{:7.4f}".format(param.value)
                tempsplit = line.split("-")
                leng = len(tempsplit)
                AA = None
                BB = None
                if leng == 2:
                    # Bond
                    nl = 2 + 3
                    AA = line[:nl].split("-")
                    BB = line[nl:].split()
                    atoms = "-".join([format(el, "<2") for el in AA]) + space3 * 5
                    BB[col] = value
                    const = "".join([format(el, ">12") for el in BB])
                elif leng == 3:
                    # Angle
                    nl = 2 + 3 * 2
                    AA = line[:nl].split("-")
                    BB = line[nl:].split()
                    atoms = "-".join([format(el, "<2") for el in AA]) + space3 * 4
                    BB[col] = value
                    const = "".join([format(el, ">12") for el in BB])
                elif leng >= 4:
                    # Dihedral/Improper
                    nl = 2 + 3 * 3
                    AA = line[:nl].split("-")
                    BB = line[nl:].split()
                    atoms = "-".join([format(el, "<2") for el in AA]) + space3 * 2
                    value = "{:7.5f}".format(param.value)
                    if param.ptype == "imp1":
                        atoms += space3
                        BB[0] = value
                        const = (
                            "".join([format(el, ">12") for el in BB[:3]])
                            + space3
                            + " ".join(BB[3:])
                        )
                    else:
                        atoms += format(BB[0], ">3")
                        # Dihedral
                        BB[1] = value
                        const = (
                            "".join([format(el, ">12") for el in BB[1:4]])
                            + space3
                            + " ".join(BB[4:])
                        )

                lines[param.mm3_row - 1] = atoms + const + "\n"
        with open(path, "w") as f:
            f.writelines(lines)
        logger.log(10, "WROTE: {}".format(path))


class MM3(FF):
    """
    Class for Schrodinger MM3* force fields (mm3.fld). TODO

    Attributes
    ----------
    smiles : list of strings
             MM3* SMILES syntax used in a custom parameter section of a
             Schrodinger MM3* force field file.
    sub_names : list of strings
                Strings used to describe each custom parameter section read.
    atom_types : list of strings
                 Atom types derived from the SMILES formula. The smiles
                 formula may have some integers, but this is strictly atom
                 types.
    lines : list of strings
            Every line from the MM3* force field file.
    """

    units = co.MM3FF

    __slots__ = ["smiles", "sub_names", "_atom_types", "_lines"]

    def __init__(self, path=None, data=None, method=None, params:List[Param]=None, score=None):
        super(MM3, self).__init__(path, data, method, params, score)
        self.smiles = []
        self.sub_names = []
        self._atom_types = None
        self._lines = None

    def copy_attributes(self, ff):
        """
        Copies some general attributes to another force field.

        Parameters
        ----------
        ff : `datatypes.MM3`
        """
        ff.path = self.path
        ff.smiles = self.smiles
        ff.sub_names = self.sub_names
        ff._atom_types = self._atom_types
        ff._lines = self._lines

    @property
    def atom_types(self):
        """
        Uses the SMILES-esque substructure definition (located
        directly below the substructre's name) to determine
        the atom types.
        """
        self._atom_types = []
        for smiles in self.smiles:
            self._atom_types.append(self.convert_smiles_to_types(smiles))
        return self._atom_types

    @property
    def lines(self):
        if self._lines is None:
            with open(self.path, "r") as f:
                self._lines = f.readlines()
        return self._lines

    @lines.setter
    def lines(self, x):
        self._lines = x

    def split_smiles(self, smiles):
        """
        Uses the MM3* SMILES substructure definition (located directly below the
        substructure's name) to determine the atom types.
        """
        split_smiles = re.split(co.RE_SPLIT_ATOMS, smiles)
        # I guess this could be an if instead of while since .remove gets rid of
        # all of them, right?
        while "" in split_smiles:
            split_smiles.remove("")
        return split_smiles

    def convert_smiles_to_types(self, smiles):
        atom_types = self.split_smiles(smiles)
        atom_types = self.convert_to_types(atom_types, atom_types)
        return atom_types

    def convert_to_types(self, atom_labels, atom_types):
        """
        Takes a list of atom_labels, which may have digits instead of atom
        types, and converts it into a list of solely atom types.

        For example,
          atom_labels = [1, 2]
          atom_types  = ["Z0", "P1", "P2"]
        would return ["Z0", "P1"].

        atom_labels - List of atom labels, which can be strings like C3, H1,
                      etc. or digits like "1" or 1.
        atom_types  - List of atom types, which are only strings like C3, H1,
                      etc.
        """
        return [
            atom_types[int(x) - 1] if x.strip().isdigit() and x != "00" else x
            for x in atom_labels
        ]

    def import_ff(self, path=None, sub_search="OPT"):
        """
        Reads parameters from mm3.fld.
        """
        if path is None:
            path = self.path
        self.params:List[Param] = []
        self.smiles = []
        self.sub_names = []
        with open(path, "r") as f:
            logger.log(15, "READING: {}".format(path))
            section_sub = False
            section_smiles = False
            section_vdw = False
            for i, line in enumerate(f):
                # These lines are for parameters.
                if not section_sub and sub_search in line and line.startswith(" C"):
                    matched = re.match("\sC\s+({})\s+".format(co.RE_SUB), line)
                    assert (
                        matched is not None
                    ), "[L{}] Can't read substructure name: {}".format(i + 1, line)
                    if matched != None:
                        # Oh good, you found your substructure!
                        section_sub = True
                        sub_name = matched.group(1).strip()
                        self.sub_names.append(sub_name)
                        logger.log(
                            15,
                            "[L{}] Start of substructure: {}".format(i + 1, sub_name),
                        )
                        section_smiles = True
                        continue
                elif section_smiles is True:
                    matched = re.match("\s9\s+({})\s".format(co.RE_SMILES), line)
                    assert (
                        matched is not None
                    ), "[L{}] Can't read substructure SMILES: {}".format(i + 1, line)
                    smiles = matched.group(1)
                    self.smiles.append(smiles)
                    logger.log(15, "  -- SMILES: {}".format(self.smiles[-1]))
                    logger.log(
                        15, "  -- Atom types: {}".format(" ".join(self.atom_types[-1]))
                    )
                    section_smiles = False
                    continue
                # Marks the end of a substructure.
                elif section_sub and line.startswith("-3"):
                    logger.log(
                        15,
                        "[L{}] End of substructure: {}".format(i, self.sub_names[-1]),
                    )
                    section_sub = False
                    continue
                if "OPT" in line and section_vdw:
                    logger.log(
                        5,
                        "[L{}] Found Van der Waals:\n{}".format(
                            i + 1, line.strip("\n")
                        ),
                    )
                    atm = line[2:5]
                    rad = line[5:15]
                    eps = line[16:26]
                    self.params.extend(
                        (
                            ParamMM3(
                                atom_types=atm,
                                ptype="vdwr",
                                mm3_col=1,
                                mm3_row=i + 1,
                                value=float(rad),
                            ),
                            ParamMM3(
                                atom_types=atm,
                                ptype="vdwe",
                                mm3_col=2,
                                mm3_row=i + 1,
                                value=float(eps),
                            ),
                        )
                    )
                    continue
                if "OPT" in line or section_sub:
                    # Bonds.
                    if match_mm3_bond(line):
                        logger.log(
                            5, "[L{}] Found bond:\n{}".format(i + 1, line.strip("\n"))
                        )
                        if section_sub:
                            atm_lbls = [line[4:6], line[8:10]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1]
                            )
                        else:
                            atm_typs = [line[4:6], line[9:11]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="be",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="bf",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                            )
                        )
                        try:
                            self.params.append(
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="q",
                                    mm3_col=3,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[2],
                                )
                            )
                        # Some bonds parameters don't use bond dipoles.
                        except IndexError:
                            pass
                        continue
                    # Angles.
                    elif match_mm3_angle(line):
                        logger.log(
                            5, "[L{}] Found angle:\n{}".format(i + 1, line.strip("\n"))
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10], line[12:14]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1]
                            )
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11], line[14:16]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="ae",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="af",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                            )
                        )
                        continue
                    # Stretch-bends.
                    elif match_mm3_stretch_bend(line):
                        logger.log(
                            5,
                            "[L{}] Found stretch-bend:\n{}".format(
                                i + 1, line.strip("\n")
                            ),
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10], line[12:14]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1]
                            )
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11], line[14:16]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.append(
                            ParamMM3(
                                atom_labels=atm_lbls,
                                atom_types=atm_typs,
                                ptype="sb",
                                mm3_col=1,
                                mm3_row=i + 1,
                                mm3_label=line[:2],
                                value=parm_cols[0],
                            )
                        )
                        continue
                    # Torsions.
                    elif match_mm3_lower_torsion(line):
                        logger.log(
                            5,
                            "[L{}] Found torsion:\n{}".format(i + 1, line.strip("\n")),
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10], line[12:14], line[16:18]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1]
                            )
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11], line[14:16], line[19:21]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    mm3_col=3,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[2],
                                ),
                            )
                        )
                        continue
                    # Higher order torsions.
                    elif match_mm3_higher_torsion(line):
                        logger.log(
                            5,
                            "[L{}] Found higher order torsion:\n{}".format(
                                i + 1, line.strip("\n")
                            ),
                        )
                        # Will break if torsions aren't also looked up.
                        atm_lbls = self.params[-1].atom_labels
                        atm_typs = self.params[-1].atom_types
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    mm3_col=3,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[2],
                                ),
                            )
                        )
                        continue
                    # Improper torsions.
                    elif match_mm3_improper(line):
                        logger.log(
                            5,
                            "[L{}] Found torsion:\n{}".format(i + 1, line.strip("\n")),
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10], line[12:14], line[16:18]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1]
                            )
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11], line[14:16], line[19:21]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="imp1",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="imp2",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                            )
                        )
                        continue
                    # Bonds.
                    elif match_mm3_vdw(line):
                        logger.log(
                            5, "[L{}] Found vdw:\n{}".format(i + 1, line.strip("\n"))
                        )
                        if section_sub:
                            atm_lbls = [line[4:6], line[8:10]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1]
                            )
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="vdwr",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[0],
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="vdwfc",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    mm3_label=line[:2],
                                    value=parm_cols[1],
                                ),
                            )
                        )
                        continue
                # The Van der Waals are stored in annoying way.
                if line.startswith("-6"):
                    section_vdw = True
                    continue
        logger.log(15, "  -- Read {} parameters.".format(len(self.params)))

    def alternate_import_ff(self, path=None, sub_search="OPT"):
        """
        Reads parameters, but doesn't need as particular of formatting.
        TODO Clean up!!!
        """
        if path is None:
            path = self.path
        self.params:List[Param] = []
        self.smiles = []
        self.sub_names = []
        with open(path, "r") as f:
            logger.log(15, "READING: {}".format(path))
            section_sub = False
            section_smiles = False
            section_vdw = False
            for i, line in enumerate(f):
                cols = line.split()
                # These lines are for parameters.
                if not section_sub and sub_search in line and line.startswith(" C"):
                    matched = re.match("\sC\s+({})\s+".format(co.RE_SUB), line)
                    assert (
                        matched is not None
                    ), "[L{}] Can't read substructure name: {}".format(i + 1, line)
                    if matched:
                        # Oh good, you found your substructure!
                        section_sub = True
                        sub_name = matched.group(1).strip()
                        self.sub_names.append(sub_name)
                        logger.log(
                            15,
                            "[L{}] Start of substructure: {}".format(i + 1, sub_name),
                        )
                        section_smiles = True
                        continue
                elif section_smiles is True:
                    matched = re.match("\s9\s+({})\s".format(co.RE_SMILES), line)
                    assert (
                        matched is not None
                    ), "[L{}] Can't read substructure SMILES: {}".format(i + 1, line)
                    smiles = matched.group(1)
                    self.smiles.append(smiles)
                    logger.log(15, "  -- SMILES: {}".format(self.smiles[-1]))
                    logger.log(
                        15, "  -- Atom types: {}".format(" ".join(self.atom_types[-1]))
                    )
                    section_smiles = False
                    continue
                # Marks the end of a substructure.
                elif section_sub and line.startswith("-3"):
                    logger.log(
                        15,
                        "[L{}] End of substructure: {}".format(i, self.sub_names[-1]),
                    )
                    section_sub = False
                    continue
                # Not implemented.
                # if 'OPT' in line and section_vdw:
                #     logger.log(5, '[L{}] Found Van der Waals:\n{}'.format(
                #             i + 1, line.strip('\n')))
                #     atm = line[2:5]
                #     rad = line[5:15]
                #     eps = line[16:26]
                #     self.params.extend((
                #             ParamMM3(atom_types = atm,
                #                      ptype = 'vdwr',
                #                      mm3_col = 1,
                #                      mm3_row = i + 1,
                #                      value = float(rad)),
                #             ParamMM3(atom_types = atm,
                #                      ptype = 'vdwe',
                #                      mm3_col = 2,
                #                      mm3_row = i + 1,
                #                      value = float(eps))))
                #     continue
                if "OPT" in line or section_sub:
                    # Bonds.
                    if match_mm3_bond(line):
                        logger.log(
                            5, "[L{}] Found bond:\n{}".format(i + 1, line.strip("\n"))
                        )
                        if section_sub:
                            atm_lbls = [cols[1], cols[2]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1]
                            )
                        # Not really implemented.
                        else:
                            atm_typs = [cols[1], cols[2]]
                            atm_lbls = atm_typs
                            # comment = line[COM_POS_START:].strip()
                            # self.sub_names.append(comment)
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="be",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[3]),
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="bf",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[4]),
                                ),
                            )
                        )
                        try:
                            self.params.append(
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="q",
                                    mm3_col=3,
                                    mm3_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[5]),
                                )
                            )
                        # Some bonds parameters don't use bond dipoles.
                        except IndexError:
                            pass
                        continue
                    # Angles.
                    elif match_mm3_angle(line):
                        logger.log(
                            5, "[L{}] Found angle:\n{}".format(i + 1, line.strip("\n"))
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [cols[1], cols[2], cols[3]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1]
                            )
                        # Not implemented.
                        else:
                            pass
                            # atm_typs = [line[4:6], line[9:11],
                            #             line[14:16]]
                            # atm_lbls = atm_typs
                            # comment = line[COM_POS_START:].strip()
                            # self.sub_names.append(comment)
                        # parm_cols = line[P_1_START:P_3_END]
                        # parm_cols = map(float, parm_cols.split())
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="ae",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[4]),
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="af",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[5]),
                                ),
                            )
                        )
                        continue
                    # Stretch-bends.
                    # elif match_mm3_stretch_bend(line):
                    #     logger.log(
                    #         5, '[L{}] Found stretch-bend:\n{}'.format(
                    #             i + 1, line.strip('\n')))
                    #     if section_sub:
                    #         # Do stuff.
                    #         atm_lbls = [line[4:6], line[8:10],
                    #                     line[12:14]]
                    #         atm_typs = self.convert_to_types(
                    #             atm_lbls, self.atom_types[-1])
                    #     else:
                    #         # Do other method.
                    #         atm_typs = [line[4:6], line[9:11],
                    #                     line[14:16]]
                    #         atm_lbls = atm_typs
                    #         comment = line[COM_POS_START:].strip()
                    #         self.sub_names.append(comment)
                    #     parm_cols = line[P_1_START:P_3_END]
                    #     parm_cols = map(float, parm_cols.split())
                    #     self.params.append(
                    #         ParamMM3(atom_labels = atm_lbls,
                    #                  atom_types = atm_typs,
                    #                  ptype = 'sb',
                    #                  mm3_col = 1,
                    #                  mm3_row = i + 1,
                    #                  mm3_label = line[:2],
                    #                  value = parm_cols[0]))
                    #     continue
                    # Torsions.
                    elif match_mm3_lower_torsion(line):
                        logger.log(
                            5,
                            "[L{}] Found torsion:\n{}".format(i + 1, line.strip("\n")),
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [cols[1], cols[2], cols[3], cols[4]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1]
                            )
                        else:
                            pass
                            # Do other method.
                            # atm_typs = [line[4:6], line[9:11],
                            #             line[14:16], line[19:21]]
                            # atm_lbls = atm_typs
                            # comment = line[COM_POS_START:].strip()
                            # self.sub_names.append(comment)
                        # parm_cols = line[P_1_START:P_3_END]
                        # parm_cols = map(float, parm_cols.split())
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[5]),
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[6]),
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="df",
                                    mm3_col=3,
                                    mm3_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[7]),
                                ),
                            )
                        )
                        continue
                    # Higher order torsions.
                    # """elif match_mm3_higher_torsion(line):
                    #     logger.log(
                    #         5, '[L{}] Found higher order torsion:\n{}'.format(
                    #             i + 1, line.strip('\n')))
                    #     # Will break if torsions aren't also looked up.
                    #     atm_lbls = self.params[-1].atom_labels
                    #     atm_typs = self.params[-1].atom_types
                    #     parm_cols = line[P_1_START:P_3_END]
                    #     parm_cols = map(float, parm_cols.split())
                    #     self.params.extend((
                    #         ParamMM3(atom_labels = atm_lbls,
                    #                  atom_types = atm_typs,
                    #                  ptype = 'df',
                    #                  mm3_col = 1,
                    #                  mm3_row = i + 1,
                    #                  mm3_label = cols[0],
                    #                  value = parm_cols[0]),
                    #         ParamMM3(atom_labels = atm_lbls,
                    #                  atom_types = atm_typs,
                    #                  ptype = 'df',
                    #                  mm3_col = 2,
                    #                  mm3_row = i + 1,
                    #                  mm3_label = cols[0],
                    #                  value = parm_cols[1]),
                    #         ParamMM3(atom_labels = atm_lbls,
                    #                  atom_types = atm_typs,
                    #                  ptype = 'df',
                    #                  mm3_col = 3,
                    #                  mm3_row = i + 1,
                    #                  mm3_label = cols[0],
                    #                  value = parm_cols[2])))
                    #     continue"""
                    # Improper torsions.
                    elif match_mm3_improper(line):
                        logger.log(
                            5,
                            "[L{}] Found torsion:\n{}".format(i + 1, line.strip("\n")),
                        )
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [cols[1], cols[2], cols[3], cols[4]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1]
                            )
                        else:
                            pass
                            # Do other method.
                            # atm_typs = [line[4:6], line[9:11],
                            #             line[14:16], line[19:21]]
                            # atm_lbls = atm_typs
                            # comment = line[COM_POS_START:].strip()
                            # self.sub_names.append(comment)
                        # parm_cols = line[P_1_START:P_3_END]
                        # parm_cols = map(float, parm_cols.split())
                        self.params.extend(
                            (
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="imp1",
                                    mm3_col=1,
                                    mm3_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[5]),
                                ),
                                ParamMM3(
                                    atom_labels=atm_lbls,
                                    atom_types=atm_typs,
                                    ptype="imp2",
                                    mm3_col=2,
                                    mm3_row=i + 1,
                                    mm3_label=cols[0],
                                    value=float(cols[6]),
                                ),
                            )
                        )
                        continue
                # The Van der Waals are stored in annoying way.
                if line.startswith("-6"):
                    section_vdw = True
                    continue
        logger.log(15, "  -- Read {} parameters.".format(len(self.params)))

    def export_ff(self, path=None, params=None, lines=None):
        """
        Exports the force field to a file, typically mm3.fld.

        Parameters
        ----------
        path : string
               File to be written or overwritten.
        params : list of `datatypes.Param` (or subclass)
        lines : list of strings
                This is what is generated when you read mm3.fld using
                readlines().
        """
        if path is None:
            path = self.path
        if params is None:
            params:List[Param] = self.params
        if lines is None:
            lines = self.lines
        for param in params:
            logger.log(1, ">>> param: {} param.value: {}".format(param, param.value))
            line = lines[param.mm3_row - 1]
            # There are some problems with this. Probably an optimization
            # technique gave you these crazy parameter values. Ideally, this
            # entire trial FF should be discarded.
            # Someday export_ff should raise an exception when these values
            # get too rediculous, and this exception should be handled by the
            # optimization techniques appropriately.
            if abs(param.value) > 999.0:
                logger.warning("Value of {} is too high! Skipping write.".format(param))
            elif param.mm3_col == 1:
                lines[param.mm3_row - 1] = (
                    line[:P_1_START] + "{:10.4f}".format(param.value) + line[P_1_END:]
                )
            elif param.mm3_col == 2:
                lines[param.mm3_row - 1] = (
                    line[:P_2_START] + "{:10.4f}".format(param.value) + line[P_2_END:]
                )
            elif param.mm3_col == 3:
                lines[param.mm3_row - 1] = (
                    line[:P_3_START] + "{:10.4f}".format(param.value) + line[P_3_END:]
                )
        with open(path, "w") as f:
            f.writelines(lines)
        logger.log(10, "WROTE: {}".format(path))

    def alternate_export_ff(self, path=None, params=None):
        """
        Doesn't rely upon needing to read an mm3.fld.
        """
        lines = []
        for param in params:
            pass


def match_mm3_label(mm3_label):
    """
    Makes sure the MM3* label is recognized.

    The label is the 1st 2 characters in the line containing the parameter
    in a Schrodinger mm3.fld file.
    """
    return re.match("[\s5a-z][1-5]", mm3_label)


def match_mm3_vdw(mm3_label):
    """Matches MM3* label for bonds."""
    return re.match("[\sa-z]6", mm3_label)


def match_mm3_bond(mm3_label):
    """Matches MM3* label for bonds."""
    return re.match("[\sa-z]1", mm3_label)


def match_mm3_angle(mm3_label):
    """Matches MM3* label for angles."""
    return re.match("[\sa-z]2", mm3_label)


def match_mm3_stretch_bend(mm3_label):
    """Matches MM3* label for stretch-bends."""
    return re.match("[\sa-z]3", mm3_label)


def match_mm3_torsion(mm3_label):
    """Matches MM3* label for all orders of torsional parameters."""
    return re.match("[\sa-z]4|54", mm3_label)


def match_mm3_lower_torsion(mm3_label):
    """Matches MM3* label for torsions (1st through 3rd order)."""
    return re.match("[\sa-z]4", mm3_label)


def match_mm3_higher_torsion(mm3_label):
    """Matches MM3* label for torsions (4th through 6th order)."""
    return re.match("54", mm3_label)


def match_mm3_improper(mm3_label):
    """Matches MM3* label for improper torsions."""
    return re.match("[\sa-z]5", mm3_label)


def mass_weight_hessian(hess, atoms, reverse=False):
    """Mass weights Hessian by multiplying my 1/sqrt(mass1 * mass2). If reverse is True,
     it un-mass weights the Hessian. Note that this does not return a new object but rather
     modifies the one passed as hess.

    Args:
        hess (_type_): Hessian matrix to mass-weight, modifies the variable itself.
        atoms (_type_): Atom objects related to the Hessian (must be in correct order).
        reverse (bool, optional): Whether to reverse mass-weight (* sqrt(mass1 * mass2)). Defaults to False.
    """
    masses = [co.MASSES[x.element] for x in atoms if not x.is_dummy]
    changes = []
    for mass in masses:
        changes.extend([1 / np.sqrt(mass)] * 3)
    x, y = hess.shape
    for i in range(0, x):
        for j in range(0, y):
            if reverse:
                hess[i, j] = hess[i, j] / changes[i] / changes[j]
            else:
                hess[i, j] = hess[i, j] * changes[i] * changes[j]


def mass_weight_force_constant(
    force_const: float, atoms: List[Atom], reverse: bool = False, rm: bool = False
) -> float:
    """Mass weights force constant. If reverse is True, it un-mass weights
    the force constant.

    Args:
        force_const (float): force constant value to mass-weight or un-mass-weight.
        atoms (List[Atom]): Atoms associated with the force constant.
        reverse (bool, optional): Whether to un-mass-weight the force constant instead. Defaults to False.
        rm (bool, optional): Whether to instead convert the force constant to reduced mass representation. Defaults to False.

    Returns:
        float: mass-weighted or un-mass-weighted value of force constant.
    """
    force_constant = force_const
    masses = [co.MASSES[x.element] for x in atoms]
    changes = []
    if rm:
        return force_constant * np.sqrt(masses[0] + masses[1])
    for mass in masses:
        change = 1 / np.sqrt(mass)
        if reverse:
            force_constant = force_constant / change
        else:
            force_constant = force_constant * change
    return force_constant


def mass_weight_eigenvectors(evecs, atoms, reverse=False):
    """
    Mass weights eigenvectors. If reverse is True, it un-mass weights
    the eigenvectors. TODO
    """
    changes = []
    for atom in atoms:
        if not atom.is_dummy:
            changes.extend([np.sqrt(atom.exact_mass)] * 3)
    x, y = evecs.shape
    for i in range(0, x):
        for j in range(0, y):
            if reverse:
                evecs[i, j] /= changes[j]
            else:
                evecs[i, j] *= changes[j]
