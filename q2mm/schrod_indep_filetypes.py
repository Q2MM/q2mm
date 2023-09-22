from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import logging
import numpy as np
import os
import re
import sys

import constants as co

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

# Print out full matrices rather than having Numpy truncate them.
# np.nan seems to no longer be supported for untruncated printing
# of arrays. The suggestion is to use sys.maxsize but I haven't checked
# that this works for python2 so leaving the commented code for now.
# np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=sys.maxsize)

class Structure(object):
    """
    Data for a single structure/conformer/snapshot.
    """
    __slots__ = ['atoms', 'bonds', 'angles', 'torsions', 'hess', 'props']
    def __init__(self):
        self.atoms = []
        self.bonds = []
        self.angles = []
        self.torsions = []
        self.hess = None
        self.props = {}
    @property
    def coords(self):
        """
        Returns atomic coordinates as a list of lists.
        """
        return [atom.coords for atom in self.atoms]
    def format_coords(self, format='latex', indices_use_charge=None):
        """
        Returns a list of strings/lines to easily generate coordinates
        in various formats.

        latex  - Makes a LaTeX table.
        gauss  - Makes output that matches Gaussian's .com filse.
        jaguar - Just like Gaussian, but include the atom number after the
                 element name in the left column.
        """
        # Formatted for LaTeX.
        if format == 'latex':
            output = ['\\begin{tabular}{l S[table-format=3.6] '
                      'S[table-format=3.6] S[table-format=3.6]}']
            for i, atom in enumerate(self.atoms):
                if atom.element is None:
                    ele = co.MASSES.items()[atom.atomic_num - 1][0]
                else:
                    ele = atom.element
                output.append('{0}{1} & {2:3.6f} & {3:3.6f} & '
                              '{4:3.6f}\\\\'.format(
                        ele, i+1, atom.x, atom.y, atom.z))
            output.append('\\end{tabular}')
            return output
        # Formatted for Gaussian .com's.
        elif format == 'gauss':
            output = []
            for i, atom in enumerate(self.atoms):
                if atom.element is None:
                    ele = co.MASSES.items()[atom.atomic_num - 1][0]
                else:
                    ele = atom.element
                # Used only for a problem Eric experienced.
                # if ele == '': ele = 'Pd'
                if indices_use_charge:
                    if atom.index in indices_use_charge:
                        output.append(
                            ' {0:s}--{1:.5f}{2:>16.6f}{3:16.6f}'
                            '{4:16.6f}'.format(
                                ele, atom.partial_charge, atom.x,
                                atom.y, atom.z))
                    else:
                        output.append(' {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}'.format(
                                ele, atom.x, atom.y, atom.z))
                else:
                    output.append(' {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}'.format(
                            ele, atom.x, atom.y, atom.z))
            return output
        # Formatted for Jaguar.
        elif format == 'jaguar':
            output = []
            for i, atom in enumerate(self.atoms):
                if atom.element is None:
                    ele = co.MASSES.items()[atom.atomic_num - 1][0]
                else:
                    ele = atom.element
                # Used only for a problem Eric experienced.
                # if ele == '': ele = 'Pd'
                label = '{}{}'.format(ele, atom.index)
                output.append(' {0:<8s}{1:>16.6f}{2:>16.6f}{3:>16.6f}'.format(
                        label, atom.x, atom.y, atom.z))
            return output
    def select_stuff(self, typ, com_match=None):
        """
        A much simpler version of select_data. It would be nice if select_data
        was a wrapper around this function.
        """
        stuff = []
        for thing in getattr(self, typ):
            if (com_match and any(x in thing.comment for x in com_match)) or \
                    com_match is None:
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
        logger.log(1, '>>> typ: {}'.format(typ))
        for thing in getattr(self, typ):
            if (com_match and any(x in thing.comment for x in com_match)) or \
                    com_match is None:
                datum = thing.as_data(**kwargs)
                # If it's a torsion we have problems.
                # Have to check whether an angle inside the torsion is near 0 or 180.
                if typ == 'torsions':
                    atom_nums = [datum.atm_1, datum.atm_2, datum.atm_3, datum.atm_4]
                    angle_atoms_1 = [atom_nums[0], atom_nums[1], atom_nums[2]]
                    angle_atoms_2 = [atom_nums[1], atom_nums[2], atom_nums[3]]
                    for angle in self.angles:
                        if set(angle.atom_nums) == set(angle_atoms_1):
                            angle_1 = angle.value
                            break
                    for angle in self.angles:
                        if set(angle.atom_nums) == set(angle_atoms_2):
                            angle_2 = angle.value
                            break
                    try:
                        logger.log(1, '>>> atom_nums: {}'.format(atom_nums))
                        logger.log(1, '>>> angle_1: {} / angle_2: {}'.format(
                                angle_1, angle_2))
                    except UnboundLocalError:
                        logger.error('>>> atom_nums: {}'.format(atom_nums))
                        logger.error(
                            '>>> angle_atoms_1: {}'.format(angle_atoms_1))
                        logger.error(
                            '>>> angle_atoms_2: {}'.format(angle_atoms_2))
                        if 'angle_1' not in locals():
                            logger.error("Can't identify angle_1!")
                        else:
                            logger.error(">>> angle_1: {}".format(angle_1))
                        if 'angle_2' not in locals():
                            logger.error("Can't identify angle_2!")
                        else:
                            logger.error(">>> angle_2: {}".format(angle_2))
                        logger.warning('WARNING: Using torsion anyway!')
                        data.append(datum)
                    if -20. < angle_1 < 20. or 160. < angle_1 < 200. or \
                            -20. < angle_2 < 20. or 160. < angle_2 < 200.:
                        logger.log(
                            1, '>>> angle_1 or angle_2 is too close to 0 or 180!')
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
        for atom in self.atoms:
            if 40 < atom.atom_type < 49:
                for bonded_atom_index in atom.bonded_atom_indices:
                    bonded_atom = self.atoms[bonded_atom_index - 1]
                    if bonded_atom.atom_type == 3:
                        aliph_hyds.append(atom)
        logger.log(5, '  -- {} aliphatic hydrogen(s).'.format(len(aliph_hyds)))
        return aliph_hyds
    def get_hyds(self):
        """
        Returns the atom numbers of any default MacroModel type hydrogens.

        This should be subclassed into something is MM3* specific.
        """
        hyds = []
        for atom in self.atoms:
            if 40 < atom.atom_type < 49:
                for bonded_atom_index in atom.bonded_atom_indices:
                    hyds.append(atom)
        logger.log(5, '  -- {} hydrogen(s).'.format(len(hyds)))
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
        for atom in self.atoms:
            if atom.is_dummy:
                logger.log(
                    10,'  -- Identified {} as a dummy atom.'.format(atom))
                dummies.append(atom.index)
        return dummies

class Atom(object):
    """
    Data class for a single atom.

    Really, some of this atom type stuff should perhaps be in a MM3*
    specific atom class.
    """
    __slots__ = ['atom_type', 'atom_type_name', 'atomic_num', 'atomic_mass',
                 'bonded_atom_indices', 'coords_type', '_element',
                 '_exact_mass', 'index', 'partial_charge', 'x', 'y', 'z',
                 'props']
    def __init__(self, atom_type=None, atom_type_name=None, atomic_num=None,
                 atomic_mass=None, bonded_atom_indices=None, coords=None,
                 coords_type=None, element=None, exact_mass=None, index=None,
                 partial_charge=None, x=None, y=None, z=None):
        self.atom_type = atom_type
        self.atom_type_name = atom_type_name
        self.atomic_num = atomic_num
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
        if coords:
            self.x = coords[0]
            self.y = coords[1]
            self.z = coords[2]
        self.props = {}
    def __repr__(self):
            return '{}[{},{},{}]'.format(
                self.atom_type_name, self.x, self.y, self.z)
    @property
    def coords(self):
        return [self.x, self.y, self.z]
    @coords.setter
    def coords(self, value):
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
    # I have no idea if these atom types are actually correct.
    # Really, the user should specify custom atom types, such as dummies, in a
    # configuration file somewhere.
    @property
    def is_dummy(self):
        """
        Return True if self is a dummy atom, else return False.

        Returns
        -------
        bool
        """
        # I think 61 is the default dummy atom type in a Schrodinger atom.typ
        # file.
        # Okay, so maybe it's not. Anyway, Tony added an atom type 205 for
        # dummies. It'd be really great if we all used the same atom.typ file
        # someday.
        # Could add in a check for the atom_type number. I removed it.
        if self.atom_type_name == 'Du' or \
                self.element == 'X' or \
                self.atomic_num == -2:
            return True
        else:
            return False

class Bond(object):
    """
    Data class for a single bond.
    """
    __slots__ = ['atom_nums', 'comment', 'order', 'value', 'ff_row']
    def __init__(self, atom_nums=None, comment=None, order=None, value=None,
                 ff_row=None):
        self.atom_nums = atom_nums
        self.comment = comment
        self.order = order
        self.value = value
        self.ff_row = ff_row
    def __repr__(self):
        return '{}[{}]({})'.format(
            self.__class__.__name__, '-'.join(
                map(str, self.atom_nums)), self.value)
    def as_data(self, **kwargs):
        # Sort of silly to have all this stuff about angles and
        # torsions in here, but they both inherit from this class.
        # I suppose it'd make more sense to create a structural
        # element class that these all inherit from.
        # Warning that I recently changed these labels, and that
        # may have consequences.
        if self.__class__.__name__.lower() == 'bond':
            typ = 'b'
        elif self.__class__.__name__.lower() == 'angle':
            typ = 'a'
        elif self.__class__.__name__.lower() == 'torsion':
            typ = 't'
        datum = Datum(val=self.value, typ=typ,ff_row=self.ff_row)
        for i, atom_num in enumerate(self.atom_nums):
            setattr(datum, 'atm_{}'.format(i+1), atom_num)
        for k, v in kwargs.items():
            setattr(datum, k, v)
        return datum

class Angle(Bond):
    """
    Data class for a single angle.
    """
    def __init__(self, atom_nums=None, comment=None, order=None, value=None,
                 ff_row=None):
        super(Angle, self).__init__(atom_nums, comment, order, value, ff_row)

class Torsion(Bond):
    """
    Data class for a single torsion.
    """
    def __init__(self, atom_nums=None, comment=None, order=None, value=None,
                 ff_row=None):
        super(Torsion, self).__init__(atom_nums, comment, order, value, ff_row)

class File(object):
    """
    Base for every other filetype class.
    """
    def __init__(self, path):
        self._lines = None
        self.path = os.path.abspath(path)
        # self.path = path
        self.directory = os.path.dirname(self.path)
        self.filename = os.path.basename(self.path)
        # self.name = os.path.splitext(self.filename)[0]
    @property
    def lines(self):
        if self._lines is None:
            with open(self.path, 'r') as f:
                self._lines = f.readlines()
        return self._lines
    def write(self, path, lines=None):
        if lines is None:
            lines = self.lines
        with open(path, 'w') as f:
            for line in lines:
                f.write(line)


class GaussLog(File):
    """
    Used to retrieve data from Gaussian log files.

    If you are extracting frequencies/Hessian data from this file, use
    the keyword NoSymmetry when running the Gaussian calculation.
    """
    def __init__(self, path):
        super(GaussLog, self).__init__(path)
        self._evals = None
        self._evecs = None
        self._structures = None
        self._esp_rms = None
    @property
    def evecs(self):
        if self._evecs is None:
            self.read_out()
        return self._evecs
    @property
    def evals(self):
        if self._evals is None:
            self.read_out()
        return self._evals
    @property
    def structures(self):
        if self._structures is None:
            #self.read_out()
            self.read_archive()
        return self._structures
    @property
    def esp_rms(self):
        if self._esp_rms is None:
            self._esp_rms = -1
            self.read_out()
        return self._esp_rms
    def read_out(self):
        """
        Read force constant and eigenvector data from a frequency
        calculation.
        """
        logger.log(5, 'READING: {}'.format(self.filename))
        self._evals = []
        self._evecs = []
        self._structures = []
        force_constants = []
        evecs = []
        with open(self.path, 'r') as f:
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
                if 'Charges from ESP fit' in line:
                    pattern = re.compile('RMS=\s+({0})'.format(co.RE_FLOAT))
                    match = pattern.search(line)
                    self._esp_rms = float(match.group(1))
                # Gathering some geometric information.
                elif 'Standard orientation:' in line:
                    self._structures.append(Structure())
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    next(file_iterator)
                    line = next(file_iterator)
                    while not '---' in line:
                        cols = line.split()
                        self._structures[-1].atoms.append(
                            Atom(index=int(cols[0]),
                                 atomic_num=int(cols[1]),
                                 x=float(cols[3]),
                                 y=float(cols[4]),
                                 z=float(cols[5])))
                        line = next(file_iterator)
                    logger.log(5, '  -- Found {} atoms.'.format(
                            len(self._structures[-1].atoms)))
                elif 'Harmonic' in line:
                    # The high quality eigenvectors come before the low quality
                    # ones. If you see "Harmonic" again, it means you're at the
                    # low quality ones now, so break.
                    if past_first_harm:
                        break
                    else:
                        past_first_harm = True
                elif 'Frequencies' in line:
                    # We're going to keep reusing these.
                    # We accumulate sets of eigevectors and eigenvalues, add
                    # them to self._evecs and self._evals, and then reuse this
                    # for the next set.
                    del(force_constants[:])
                    del(evecs[:])
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
                        if frequency < 0.:
                            force_constants.append(-1.)
                        else:
                            force_constants.append(1.)
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
                    if 'Coord' in line:
                        hpmodes = True
                    # This is different depending on whether you use
                    # "freq=projected".
                    line = next(file_iterator)
                    # The "projected" keyword seems to add "IRC Coupling".
                    if 'IRC Coupling' in line:
                        line = next(file_iterator)
                    # We're on to the eigenvectors.
                    # Until the end of this section containing the eigenvectors,
                    # the number of columns remains constant. When that changes,
                    # we know we're to the next set of frequencies, force
                    # constants and eigenvectors.
                    cols = line.split()
                    cols_len = len(cols)

                    while len(cols) == cols_len:
                        # This will come after all the eigenvectors have been
                        # read. We can break out then.
                        if 'Harmonic' in line:
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
                        mass_sqrt = np.sqrt(list(co.MASSES.items())[int(cols[1]) - 1][1])

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
                    if 'Harmonic' in line:
                        break
        if self._evals and self._evecs:
            for evec in self._evecs:
                # Each evec is a single eigenvector.
                # Add up the sum of squares over an eigenvector.
                sum_of_squares = 0.
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
            logger.log(1, '>>> self._evals: {}'.format(self._evals))
            logger.log(1, '>>> self._evecs: {}'.format(self._evecs))
            logger.log(5, '  -- {} structures found.'.format(
                len(self.structures)))
    # May want to move some attributes assigned to the structure class onto
    # this filetype class.
    def read_archive(self):
        """
        Only reads last archive found in the Gaussian .log file.
        """
        logger.log(5, 'READING: {}'.format(self.filename))
        struct = Structure()
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
                '(?s)(\s1\\\\1\\\\.*?[\\\\\n\s]+@)',
                open(self.path, 'r').read())[-1]
            logger.log(5, '  -- Located last archive.')
        except IndexError:
            logger.warning("  -- Couldn't locate archive.")
            raise
        # Make it into one string.
        arch = arch.replace('\n ', '')
        # Separate it by Gaussian's section divider.
        arch = arch.split('\\\\')
        # Helps us iterate over sections of the archive.
        section_counter = 0
        # SECTION 0
        # General job information.
        arch_general = arch[section_counter]
        section_counter += 1
        stuff = re.search(
            '\s1\\\\1\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\.*?\\\\(?P<user>.*?)'
            '\\\\(?P<date>.*?)'
            '\\\\.*?',
            arch_general)
        struct.props['user'] = stuff.group('user')
        struct.props['date'] = stuff.group('date')
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
        section_counter +=1
        stuff = re.search(
            '(?P<charge>.*?)'
            ',(?P<multiplicity>.*?)'
            '\\\\(?P<atoms>.*)',
            arch_coords)
        struct.props['charge'] = stuff.group('charge')
        struct.props['multiplicity'] = stuff.group('multiplicity')
        # We want to do more fancy stuff with the atoms than simply add to
        # the properties dictionary.
        atoms = stuff.group('atoms')
        atoms = atoms.split('\\')
        # Z-matrix coordinates adds another section. We need to be aware of
        # this.
        probably_z_matrix = False
        for atom in atoms:
            stuff = atom.split(',')
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
                    'Not sure how to read coordinates from Gaussian acrhive!')
                probably_z_matrix = True
                section_counter += 1
                # Let's have it stop looping over atoms, but not fail anymore.
                break
                # raise Exception(
                #     'Not sure how to read coordinates from Gaussian archive!')
            struct.atoms.append(
                Atom(element=ele, x=float(x), y=float(y), z=float(z)))
        logger.log(20, '  -- Read {} atoms.'.format(len(struct.atoms)))
        # SECTION 4
        # All sorts of information here. This area looks like:
        #     prop1=value1\prop2=value2\prop3=value3
        arch_info = arch[section_counter]
        section_counter += 1
        arch_info = arch_info.split('\\')
        for thing in arch_info:
            prop_name, prop_value = thing.split('=')
            struct.props[prop_name] = prop_value
        # SECTION 5
        # The Hessian. Only exists if you did a frequency calculation.
        # Appears in lower triangular form.
        if not arch[section_counter] == '@':
            hess_tri = arch[section_counter]
            hess_tri = hess_tri.split(',')
            logger.log(
                5,
                '  -- Read {} Hessian elements in lower triangular '
                'form.'.format(len(hess_tri)))
            hess = np.zeros([len(atoms) * 3, len(atoms) * 3], dtype=float)
            logger.log(
                5, '  -- Created {} Hessian matrix.'.format(hess.shape))
            # Code for if it was in upper triangle (it's not).
            # hess[np.triu_indices_from(hess)] = hess_tri
            # hess += np.triu(hess, -1).T
            # Lower triangle code.
            hess[np.tril_indices_from(hess)] = hess_tri
            hess += np.tril(hess, -1).T
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
        fields = ['RMS Force', 'RMS Displacement', 'Maximum Force',
                  'Maximum Displacement']
        for i, structure in reversed(list(enumerate(structures))):
            yes_or_no = [value[2] for key, value in structure.props.items()
                         if key in fields]
            if not structure.atoms:
                logger.warning('  -- No atoms found in structure {}. '
                               'Skipping.'.format(i+1))
                continue
            if len(yes_or_no) == 4:
                structures_compared += 1
                if best_structure is None:
                    logger.log(10, '  -- Most converged structure: {}'.format(
                            i+1))
                    best_structure = structure
                    best_yes_or_no = yes_or_no
                elif yes_or_no.count('YES') > best_yes_or_no.count('YES'):
                    best_structure = structure
                    best_yes_or_no = yes_or_no
                elif yes_or_no.count('YES') == best_yes_or_no.count('YES'):
                    number_better = 0
                    for field in fields:
                        if structure.props[field][0] < \
                                best_structure.props[field][0]:
                            number_better += 1
                    if number_better > 2:
                        best_structure = structure
                        best_yes_or_no = yes_or_no
            elif len(yes_or_no) != 0:
                logger.warning(
                    '  -- Partial convergence criterion in structure: {}'.format(
                        self.path))
        logger.log(10, '  -- Compared {} out of {} structures.'.format(
                structures_compared, len(self.structures)))
        return best_structure
    def read_any_coords(self, coords_type='both'):
        logger.log(10, 'READING: {}'.format(self.filename))
        structures = []
        with open(self.path, 'r') as f:
            section_coords_input = False
            section_coords_standard = False
            section_convergence = False
            section_optimization = False
            for i, line in enumerate(f):
                    # Look for input coordinates.
                    if coords_type == 'input' or coords_type == 'both':
                        # Marks end of input coords for a given structure.
                        if section_coords_input and 'Distance matrix' in line:
                            section_coords_input = False
                            logger.log(5, '[L{}] End of input coordinates '
                                       '({} atoms).'.format(
                                    i+1, count_atom))
                        # Add atoms and coordinates to structure.
                        if section_coords_input:
                            match = re.match(
                                '\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+({0})\s+'
                                '({0})'.format(co.RE_FLOAT), line)
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1))-1]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num:
                                    assert current_atom.atomic_num == int(
                                        match.group(2)), \
                                        ("[L{}] Atomic numbers don't match "
                                         "(current != existing) "
                                         "({} != {}).".format(
                                                i+1, int(match.group(2)),
                                                current_atom.atomic_num))
                                else:
                                    current_atom.atomic_num = int(
                                        match.group(2))
                                current_atom.index = int(match.group(1))
                                current_atom.coords_type = 'input'
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # Start of input coords for a given structure.
                        if not section_coords_input and \
                                'Input orientation:' in line:
                            current_structure = Structure()
                            structures.append(current_structure)
                            section_coords_input = True
                            count_atom = 0
                            logger.log(5, '[L{}] Start input coordinates '
                                       'section.'.format(i+1))
                    # Look for standard coordinates.
                    if coords_type == 'standard' or coords_type == 'both':
                        # End of coordinates for a given structure.
                        if section_coords_standard and \
                                ('Rotational constants' in line or
                                 'Leave Link' in line):
                            section_coords_standard = False
                            logger.log(5, '[L{}] End standard coordinates '
                                       'section ({} atoms).'.format(
                                    i+1, count_atom))
                        # Grab coordinates for each atom.
                        # Add atoms to the structure.
                        if section_coords_standard:
                            match = re.match('\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+'
                                             '({0})\s+({0})'.format(
                                    co.RE_FLOAT), line)
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1))-1]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num:
                                    assert current_atom.atomic_num == int(
                                        match.group(2)), \
                                        ("[L{}] Atomic numbers don't match "
                                         "(current != existing) "
                                         "({} != {}).".format(
                                                i+1, int(match.group(2)),
                                                current_atom.atomic_num))
                                else:
                                    current_atom.atomic_num = int(
                                        match.group(2))
                                current_atom.index = int(match.group(1))
                                current_atom.coords_type = 'standard'
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # Start of standard coordinates.
                        if not section_coords_standard and \
                                'Standard orientation' in line:
                            current_structure = Structure()
                            structures.append(current_structure)
                            section_coords_standard = True
                            count_atom = 0
                            logger.log(5, '[L{}] Start standard coordinates '
                                       'section.'.format(i+1))
        return structures
    def read_optimization(self, coords_type='both'):
        """
        Finds structures from a Gaussian geometry optimization that
        are listed throughout the log file. Also finds data about
        their convergence.

        coords_type = "input" or "standard" or "both"
                      Using both may cause coordinates in one format
                      to be overwritten by whatever comes later in the
                      log file.
        """
        logger.log(10, 'READING: {}'.format(self.filename))
        structures = []
        with open(self.path, 'r') as f:
            section_coords_input = False
            section_coords_standard = False
            section_convergence = False
            section_optimization = False
            for i, line in enumerate(f):
                # Look for start of optimization section of log file and
                # set a flag that it has indeed started.
                if section_optimization and 'Optimization stopped.' in line:
                    section_optimization = False
                    logger.log(5, '[L{}] End optimization section.'.format(i+1))
                if not section_optimization and \
                        'Search for a local minimum.' in line:
                    section_optimization = True
                    logger.log(5, '[L{}] Start optimization section.'.format(
                            i+1))
                if section_optimization:
                    # Start of a structure.
                    if 'Step number' in line:
                        structures.append(Structure())
                        current_structure = structures[-1]
                        logger.log(5, '[L{}] Added structure '
                                   '(currently {}).'.format(
                                i+1, len(structures)))
                    # Look for convergence information related to a single
                    # structure.
                    if section_convergence and 'GradGradGrad' in line:
                        section_convergence = False
                        logger.log(5, '[L{}] End convergence section.'.format(
                                i+1))
                    if section_convergence:
                        match = re.match(
                            '\s(Maximum|RMS)\s+(Force|Displacement)\s+({0})\s+'
                            '({0})\s+(YES|NO)'.format(
                                co.RE_FLOAT), line)
                        if match:
                            current_structure.props['{} {}'.format(
                                    match.group(1), match.group(2))] = \
                                (float(match.group(3)),
                                 float(match.group(4)), match.group(5))
                    if 'Converged?' in line:
                        section_convergence = True
                        logger.log(5, '[L{}] Start convergence section.'.format(
                                i+1))
                    # Look for input coords.
                    if coords_type == 'input' or coords_type == 'both':
                        # End of input coords for a given structure.
                        if section_coords_input and 'Distance matrix' in line:
                            section_coords_input = False
                            logger.log(5, '[L{}] End input coordinates section '
                                       '({} atoms).'.format(
                                    i+1, count_atom))
                        # Add atoms and coords to structure.
                        if section_coords_input:
                            match = re.match(
                                '\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+({0})\s+'
                                '({0})'.format(
                                    co.RE_FLOAT), line)
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1))-1]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num:
                                    assert current_atom.atomic_num == \
                                        int(match.group(2)), \
                                        ("[L{}] Atomic numbers don't match "
                                         "(current != existing) "
                                         "({} != {}).".format(
                                                i+1, int(match.group(2)),
                                                current_atom.atomic_num))
                                else:
                                    current_atom.atomic_num = \
                                        int(match.group(2))
                                current_atom.index = int(match.group(1))
                                current_atom.coords_type = 'input'
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # Start of input coords for a given structure.
                        if not section_coords_input and \
                                'Input orientation:' in line:
                            section_coords_input = True
                            count_atom = 0
                            logger.log(5, '[L{}] Start input coordinates '
                                       'section.'.format(i+1))
                    # Look for standard coords.
                    if coords_type == 'standard' or coords_type == 'both':
                        # End of coordinates for a given structure.
                        if section_coords_standard and \
                                ('Rotational constants' in line or
                                 'Leave Link' in line):
                            section_coords_standard = False
                            logger.log(5, '[L{}] End standard coordinates '
                                       'section ({} atoms).'.format(
                                    i+1, count_atom))
                        # Grab coords for each atom. Add atoms to the structure.
                        if section_coords_standard:
                            match = re.match('\s+(\d+)\s+(\d+)\s+\d+\s+({0})\s+'
                                             '({0})\s+({0})'.format(
                                    co.RE_FLOAT), line)
                            if match:
                                count_atom += 1
                                try:
                                    current_atom = current_structure.atoms[
                                        int(match.group(1))-1]
                                except IndexError:
                                    current_structure.atoms.append(Atom())
                                    current_atom = current_structure.atoms[-1]
                                if current_atom.atomic_num:
                                    assert current_atom.atomic_num == int(
                                        match.group(2)), \
                                        ("[L{}] Atomic numbers don't match "
                                         "(current != existing) "
                                         "({} != {}).".format(
                                            i+1, int(match.group(2)),
                                            current_atom.atomic_num))
                                else:
                                    current_atom.atomic_num = int(
                                        match.group(2))
                                current_atom.index = int(match.group(1))
                                current_atom.coords_type = 'standard'
                                current_atom.x = float(match.group(3))
                                current_atom.y = float(match.group(4))
                                current_atom.z = float(match.group(5))
                        # Start of standard coords.
                        if not section_coords_standard and \
                                'Standard orientation' in line:
                            section_coords_standard = True
                            count_atom = 0
                            logger.log(5, '[L{}] Start standard coordinates '
                                       'section.'.format(i+1))
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
    A single parameter.

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
    __slots__ = ['_allowed_range', '_step', '_value', 'd1', 'd2', 'ptype',
                 'simp_var']
    def __init__(self, d1=None, d2=None, ptype=None, value=None):
        self._allowed_range = None
        self._step = None
        self._value = None
        self.d1 = d1
        self.d2 = d2
        self.ptype = ptype
        self.simp_var = None
        self.value = value
    def __repr__(self):
        return '{}[{}]({:7.4f})'.format(
            self.__class__.__name__, self.ptype, self.value)
    @property
    def allowed_range(self):
        """
        Returns True or False, depending on whether the parameter is
        allowed to be negative values.
        """
        if self._allowed_range is None and self.ptype is not None:
            if self.ptype in ['q', 'df']:
                self._allowed_range = [-float('inf'), float('inf')]
            else:
                self._allowed_range = [0., float('inf')]
        return self._allowed_range
    @property
    def step(self):
        """
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
                    "provided!".format(self))
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
        if self.ptype == 'ae' and self._value > 180.:
            self._value = 180. - abs(180 - self._value)
        return self._value
    @value.setter
    def value(self, value):
        """
        When you try to give the parameter a value, make sure that's okay.
        """
        if self.value_in_range(value):
            self._value = value
    def value_in_range(self, value):
        if self.allowed_range[0] <= value <= self.allowed_range[1]:
            return True
        elif value == self.allowed_range[0] - 0.1:
            raise ParamBE("{} Backward Error. Forward Derivative only".format(str(self)))
        elif value == self.allowed_range[1] + 0.1:
            raise ParamFE("{} Forward Error. Backward Derivative only".format(str(self)))
        elif value == self.allowed_range[1] or value == self.allowed_range[0]:
            return True
        else:
            raise ParamError(
                "{} isn't allowed to have a value of {}! "
                "({} <= x <= {})".format(
                    str(self),
                    value,
                    self.allowed_range[0],
                    self.allowed_range[1]))

    def value_at_limits(self):
        # Checks if the parameter is at the limits of
        # its allowed range. Should only be run at the
        # end of an optimization to warn users they should
        # consider whether this is ok.
        if self.value == min(self.allowed_range):
            logger.warning(
                "{} is equal to its lower limit of {}!\nReconsider "
                "if you need to adjust limits, initial parameter "
                "values, or if your reference data is appropriate.".format(
                    str(self),
                    self.value))
        if self.value == max(self.allowed_range):
            logger.warning(
                "{} is equal to its upper limit of {}!\nReconsider "
                "if you need to adjust limits, initial parameter "
                "values, or if your reference data is appropriate.".format(
                    str(self),
                    self.value))

# Need a general index scheme/method/property to compare the equalness of two
# parameters, rather than having to rely on some expression that compares
# mm3_row and mm3_col.
class ParamMM3(Param):
    '''
    Adds information to Param that is specific to MM3* parameters.
    '''
    __slots__ = ['atom_labels', 'atom_types', 'mm3_col', 'mm3_row', 'mm3_label']
    def __init__(self, atom_labels=None, atom_types=None, mm3_col=None,
                 mm3_row=None, mm3_label=None,
                 d1=None, d2=None, ptype=None, value=None):
        self.atom_labels = atom_labels
        self.atom_types = atom_types
        self.mm3_col = mm3_col
        self.mm3_row = mm3_row
        self.mm3_label = mm3_label
        super(ParamMM3, self).__init__(ptype=ptype, value=value)
    def __repr__(self):
        return '{}[{}][{},{}]({})'.format(
            self.__class__.__name__, self.ptype, self.mm3_row, self.mm3_col,
            self.value)
    def __str__(self):
        return '{}[{}][{},{}]({})'.format(
            self.__class__.__name__, self.ptype, self.mm3_row, self.mm3_col,
            self.value)

class Datum(object):
    '''
    Class for a reference or calculated data point.
    '''
    __slots__ = ['_lbl', 'val', 'wht', 'typ', 'com', 'src_1', 'src_2', 'idx_1',
                 'idx_2', 'atm_1', 'atm_2', 'atm_3', 'atm_4', 'ff_row']
    def __init__(self, lbl=None, val=None, wht=None, typ=None, com=None,
                 src_1=None, src_2=None,
                 idx_1=None, idx_2=None,
                 atm_1=None, atm_2=None, atm_3=None, atm_4=None,
                 ff_row=None):
        self._lbl   = lbl
        self.val    = val
        self.wht    = wht
        self.typ    = typ
        self.com    = com
        self.src_1  = src_1
        self.src_2  = src_2
        self.idx_1  = idx_1
        self.idx_2  = idx_2
        self.atm_1  = atm_1
        self.atm_2  = atm_2
        self.atm_3  = atm_3
        self.atm_4  = atm_4
        self.ff_row = ff_row
    def __repr__(self):
        return '{}({:7.4f})'.format(self.lbl, self.val)
    @property
    def lbl(self):
        if self._lbl is None:
            a = self.typ
            if self.src_1:
                b = re.split('[.]+', self.src_1)[0]
            # Why would it ever not have src_1?
            else:
                b = None
            c = '-'.join([str(x) for x in remove_none(self.idx_1, self.idx_2)])
            d = '-'.join([str(x) for x in remove_none(
                        self.atm_1, self.atm_2, self.atm_3, self.atm_4)])
            abcd = remove_none(a, b, c, d)
            self._lbl = '_'.join(abcd)
        return self._lbl

def remove_none(*args):
    return [x for x in args if (x is not None and x is not '')]

def datum_sort_key(datum):
    '''
    Used as the key to sort a list of Datum instances. This should always ensure
    that the calculated and reference data points align properly.
    '''
    return (datum.typ, datum.src_1, datum.src_2, datum.idx_1, datum.idx_2)

class FF(object):
    """
    Class for any type of force field.

    path   - Self explanatory.
    data   - List of Datum objects.
    method - String describing method used to generate this FF.
    params - List of Param objects.
    score  - Float which is the objective function score.
    """
    def __init__(self, path=None, data=None, method=None, params=None,
                 score=None):
        self.path = path
        self.data = data
        self.method = method
        self.params = params
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
        return '{}[{}]({})'.format(
            self.__class__.__name__, self.method, self.score)

class AmberFF(FF):
    """
    STUFF TO FILL IN LATER
    """
    def __init__(self, path=None, data=None, method=None, params=None,
                 score=None):
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
            with open(self.path, 'r') as f:
                self._lines = f.readlines()
        return self._lines
    @lines.setter
    def lines(self, x):
        self._lines = x
    def import_ff(self, path=None, sub_search='OPT'):
        if path is None:
            path = self.path
        bonds = ['bond', 'bond3', 'bond4', 'bond5']
        pibonds = ['pibond', 'pibond3', 'pibond4', 'pibond5']
        angles = ['angle', 'angle3', 'angle4', 'angle5']
        torsions = ['torsion', 'torsion4', 'torsion5']
        dipoles = ['dipole', 'dipole3', 'dipole4', 'dipole5']
        self.params = []
        q2mm_sec = False
        gather_data = False
        self.sub_names = []
        count = 0
        with open(path, 'r') as f:
            logger.log(15, 'READING: {}'.format(path))
            for i, line in enumerate(f):
                split = line.split()
                if not q2mm_sec and '# Q2MM' in line:
                    q2mm_sec = True
                elif q2mm_sec and '#' in line[0]:
                    self.sub_names.append(line[1:])
                    if 'OPT' in line:
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
                        at = split[0] # need number if it matters
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
                        #A1-A2 Force Const in kcal/mol/(A**2): Eq. length in A
                        AA = line[:5].split('-')
                        BB = line[5:].split()
                        at = [AA[0],AA[1]]
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                    ptype = "bf",
                                    mm3_col = 1,
                                    mm3_row = i+1,
                                    value = float(BB[0])),
                            ParamMM3(atom_types = at,
                                    ptype = "be",
                                    mm3_col = 2,
                                    mm3_row = i+1,
                                    value = float(BB[1]))))
                    # ANGLE
                    if "DIHE" in line and count == 3:
                        count = 4
                        continue
                    elif count == 3 and "IMPR" not in line:
                        AA = line[:2+3*2].split('-')
                        BB = line[2+3*2:].split()
                        at = [AA[0],AA[1],AA[2]]
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                    ptype = 'af',
                                    mm3_col = 1,
                                    mm3_row = i + 1,
                                    value = float(BB[0])),
                            ParamMM3(atom_types = at,
                                    ptype = 'ae',
                                    mm3_col = 2,
                                    mm3_row = i + 1,
                                    value = float(BB[1]))))
                    # Dihedral
                    if "IMPR" in line and count == 4:
                        count = 5
                        continue
                    elif count == 4 and "NONB" not in line:
                        # (PK/IDIVF) * (1 + cos(PN*phi - PHASE))
                        # A4 IDIVF PK PHASE PN
                        nl = 2+3*3
                        AA = line[:nl].split('-')
                        BB = line[nl:].split()
                        at = [AA[0],AA[1],AA[2],AA[3]]
                        self.params.append(
                        ParamMM3(atom_types = at,
                                ptype = 'df',
                                mm3_col = 1,
                                mm3_row = i + 1,
                                value = float(BB[1])))

                        


                    # Improper
                    if "NONB" in line and count == 5:
                        count = 6
                        continue
                    elif count == 5:
                        nl = 2+3*3
                        AA = line[:nl].split('-')
                        BB = line[nl:].split()
                        at = [AA[0],AA[1],AA[2],AA[3]]
                        self.params.append(
                        ParamMM3(atom_types = at,
                                ptype = 'imp1',
                                mm3_col = 1,
                                mm3_row = i + 1,
                                value = float(BB[0])))
                        
                    
#                    # Hbond
#                    if "NONB" in line and count == 6:
#                        count == 7
#                        continue
#                    elif count == 6:
#                        0

                    # NONB
                    if count == 6:
                        continue
                    
                    if 'vdw' == split[0]:
                    #The first float is the vdw radius, the second has to do
                    # with homoatomic well depths and the last is a reduction
                    # factor for univalent atoms (I don't think we will need
                    # any of these except for the first one).
                        at = [split[1]]
                        self.params.append(
                            ParamMM3(atom_types = at,
                                    ptype = 'vdw',
                                    mm3_col = 1,
                                    mm3_row = i + 1,
                                    value = float(split[2])))
        logger.log(15, '  -- Read {} parameters.'.format(len(self.params)))
    def export_ff(self, path=None, params=None, lines=None):
        """
        Exports the force field to a file, typically mm3.fld.
        """
        if path is None:
            path = self.path
        if params is None:
            params = self.params
        if lines is None:
            lines = self.lines
        for param in params:
            logger.log(1, '>>> param: {} param.value: {}'.format(
                    param, param.value))
            line = lines[param.mm3_row - 1]
            if abs(param.value) > 1999.:
                logger.warning(
                    'Value of {} is too high! Skipping write.'.format(param))
            else:
                atoms = ""
                const = ""
                space3 = " " * 3
                col = int(param.mm3_col-1)
                value = '{:7.4f}'.format(param.value)
                tempsplit = line.split("-") 
                leng = len(tempsplit)
                AA = None
                BB = None
                if leng == 2:
                    # Bond
                    nl = 2 + 3
                    AA = line[:nl].split('-')
                    BB = line[nl:].split()
                    atoms = '-'.join([format(el,"<2") for el in AA]) + space3 * 5
                    BB[col] = value
                    const = ''.join([format(el,">12") for el in BB])
                elif leng == 3:
                    # Angle
                    nl = 2 + 3*2
                    AA = line[:nl].split('-')
                    BB = line[nl:].split()
                    atoms = '-'.join([format(el,"<2") for el in AA]) + space3 * 4
                    BB[col] = value
                    const = ''.join([format(el,">12") for el in BB])
                elif leng >= 4:
                    # Dihedral/Improper
                    nl = 2 + 3*3
                    AA = line[:nl].split('-')
                    BB = line[nl:].split()
                    atoms = '-'.join([format(el,"<2") for el in AA]) + space3 * 2
                    value = '{:7.5f}'.format(param.value)
                    if param.ptype == "imp1":
                        atoms += space3
                        BB[0] = value
                        const = ''.join([format(el,">12") for el in BB[:3]]) + space3 + ' '.join(BB[3:])
                    else:
                        atoms += format(BB[0],">3")
                        #Dihedral
                        BB[1] = value
                        const = ''.join([format(el,">12") for el in BB[1:4]]) + space3 + ' '.join(BB[4:])
                    
                lines[param.mm3_row - 1] = (atoms+const+'\n')
        with open(path, 'w') as f:
            f.writelines(lines)
        logger.log(10, 'WROTE: {}'.format(path))

class MM3(FF):
    """
    Class for Schrodinger MM3* force fields (mm3.fld).

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
    def __init__(self, path=None, data=None, method=None, params=None,
                 score=None):
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
            with open(self.path, 'r') as f:
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
        while '' in split_smiles:
            split_smiles.remove('')
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
        return [atom_types[int(x) - 1] if x.strip().isdigit() and
                x != '00'
                else x
                for x in atom_labels]
    def import_ff(self, path=None, sub_search='OPT'):
        """
        Reads parameters from mm3.fld.
        """
        if path is None:
            path = self.path
        self.params = []
        self.smiles = []
        self.sub_names = []
        with open(path, 'r') as f:
            logger.log(15, 'READING: {}'.format(path))
            section_sub = False
            section_smiles = False
            section_vdw = False
            for i, line in enumerate(f):
                # These lines are for parameters.
                if not section_sub and sub_search in line \
                        and line.startswith(' C'):
                    matched = re.match('\sC\s+({})\s+'.format(
                            co.RE_SUB), line)
                    assert matched is not None, \
                        "[L{}] Can't read substructure name: {}".format(
                        i + 1, line)
                    if matched != None:
                        # Oh good, you found your substructure!
                        section_sub = True
                        sub_name = matched.group(1).strip()
                        self.sub_names.append(sub_name)
                        logger.log(
                            15, '[L{}] Start of substructure: {}'.format(
                                i+1, sub_name))
                        section_smiles = True
                        continue
                elif section_smiles is True:
                    matched = re.match(
                        '\s9\s+({})\s'.format(co.RE_SMILES), line)
                    assert matched is not None, \
                        "[L{}] Can't read substructure SMILES: {}".format(
                        i + 1, line)
                    smiles = matched.group(1)
                    self.smiles.append(smiles)
                    logger.log(15, '  -- SMILES: {}'.format(
                            self.smiles[-1]))
                    logger.log(15, '  -- Atom types: {}'.format(
                            ' '.join(self.atom_types[-1])))
                    section_smiles = False
                    continue
                # Marks the end of a substructure.
                elif section_sub and line.startswith('-3'):
                    logger.log(15, '[L{}] End of substructure: {}'.format(
                            i, self.sub_names[-1]))
                    section_sub = False
                    continue
                if 'OPT' in line and section_vdw:
                    logger.log(5, '[L{}] Found Van der Waals:\n{}'.format(
                            i + 1, line.strip('\n')))
                    atm = line[2:5]
                    rad = line[5:15]
                    eps = line[16:26]
                    self.params.extend((
                            ParamMM3(atom_types = atm,
                                     ptype = 'vdwr',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     value = float(rad)),
                            ParamMM3(atom_types = atm,
                                     ptype = 'vdwe',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     value = float(eps))))
                    continue
                if 'OPT' in line or section_sub:
                    # Bonds.
                    if match_mm3_bond(line):
                        logger.log(
                            5, '[L{}] Found bond:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            atm_lbls = [line[4:6], line[8:10]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        else:
                            atm_typs = [line[4:6], line[9:11]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend((
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'be',
                                         mm3_col = 1,
                                         mm3_row = i + 1,
                                         mm3_label = line[:2],
                                         value = parm_cols[0]),
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'bf',
                                         mm3_col = 2,
                                         mm3_row = i + 1,
                                         mm3_label = line[:2],
                                         value = parm_cols[1])))
                        try:
                            self.params.append(
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'q',
                                         mm3_col = 3,
                                         mm3_row = i + 1,
                                         mm3_label = line[:2],
                                         value = parm_cols[2]))
                        # Some bonds parameters don't use bond dipoles.
                        except IndexError:
                            pass
                        continue
                    # Angles.
                    elif match_mm3_angle(line):
                        logger.log(
                            5, '[L{}] Found angle:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10],
                                        line[12:14]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11],
                                        line[14:16]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend((
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'ae',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'af',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[1])))
                        continue
                    # Stretch-bends.
                    elif match_mm3_stretch_bend(line):
                        logger.log(
                            5, '[L{}] Found stretch-bend:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10],
                                        line[12:14]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11],
                                        line[14:16]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.append(
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'sb',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[0]))
                        continue
                    # Torsions.
                    elif match_mm3_lower_torsion(line):
                        logger.log(
                            5, '[L{}] Found torsion:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10],
                                        line[12:14], line[16:18]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11],
                                        line[14:16], line[19:21]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend((
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[1]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 3,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[2])))
                        continue
                    # Higher order torsions.
                    elif match_mm3_higher_torsion(line):
                        logger.log(
                            5, '[L{}] Found higher order torsion:\n{}'.format(
                                i + 1, line.strip('\n')))
                        # Will break if torsions aren't also looked up.
                        atm_lbls = self.params[-1].atom_labels
                        atm_typs = self.params[-1].atom_types
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend((
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[1]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 3,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[2])))
                        continue
                    # Improper torsions.
                    elif match_mm3_improper(line):
                        logger.log(
                            5, '[L{}] Found torsion:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [line[4:6], line[8:10],
                                        line[12:14], line[16:18]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        else:
                            # Do other method.
                            atm_typs = [line[4:6], line[9:11],
                                        line[14:16], line[19:21]]
                            atm_lbls = atm_typs
                            comment = line[COM_POS_START:].strip()
                            self.sub_names.append(comment)
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend((
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'imp1',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'imp2',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     mm3_label = line[:2],
                                     value = parm_cols[1])))
                        continue
                    # Bonds.
                    elif match_mm3_vdw(line):
                        logger.log(
                            5, '[L{}] Found vdw:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            atm_lbls = [line[4:6], line[8:10]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        parm_cols = line[P_1_START:P_3_END]
                        parm_cols = [float(x) for x in parm_cols.split()]
                        self.params.extend((
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'vdwr',
                                         mm3_col = 1,
                                         mm3_row = i + 1,
                                         mm3_label = line[:2],
                                         value = parm_cols[0]),
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'vdwfc',
                                         mm3_col = 2,
                                         mm3_row = i + 1,
                                         mm3_label = line[:2],
                                         value = parm_cols[1])))
                        continue
                # The Van der Waals are stored in annoying way.
                if line.startswith('-6'):
                    section_vdw = True
                    continue
        logger.log(15, '  -- Read {} parameters.'.format(len(self.params)))
    def alternate_import_ff(self, path=None, sub_search='OPT'):
        """
        Reads parameters, but doesn't need as particular of formatting.
        """
        if path is None:
            path = self.path
        self.params = []
        self.smiles = []
        self.sub_names = []
        with open(path, 'r') as f:
            logger.log(15, 'READING: {}'.format(path))
            section_sub = False
            section_smiles = False
            section_vdw = False
            for i, line in enumerate(f):
                cols = line.split()
                # These lines are for parameters.
                if not section_sub and sub_search in line \
                        and line.startswith(' C'):
                    matched = re.match('\sC\s+({})\s+'.format(
                            co.RE_SUB), line)
                    assert matched is not None, \
                        "[L{}] Can't read substructure name: {}".format(
                        i + 1, line)
                    if matched:
                        # Oh good, you found your substructure!
                        section_sub = True
                        sub_name = matched.group(1).strip()
                        self.sub_names.append(sub_name)
                        logger.log(
                            15, '[L{}] Start of substructure: {}'.format(
                                i+1, sub_name))
                        section_smiles = True
                        continue
                elif section_smiles is True:
                    matched = re.match(
                        '\s9\s+({})\s'.format(co.RE_SMILES), line)
                    assert matched is not None, \
                        "[L{}] Can't read substructure SMILES: {}".format(
                        i + 1, line)
                    smiles = matched.group(1)
                    self.smiles.append(smiles)
                    logger.log(15, '  -- SMILES: {}'.format(
                            self.smiles[-1]))
                    logger.log(15, '  -- Atom types: {}'.format(
                            ' '.join(self.atom_types[-1])))
                    section_smiles = False
                    continue
                # Marks the end of a substructure.
                elif section_sub and line.startswith('-3'):
                    logger.log(15, '[L{}] End of substructure: {}'.format(
                            i, self.sub_names[-1]))
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
                if 'OPT' in line or section_sub:
                    # Bonds.
                    if match_mm3_bond(line):
                        logger.log(
                            5, '[L{}] Found bond:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            atm_lbls = [cols[1], cols[2]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
                        # Not really implemented.
                        else:
                            atm_typs = [cols[1], cols[2]]
                            atm_lbls = atm_typs
                            # comment = line[COM_POS_START:].strip()
                            # self.sub_names.append(comment)
                        self.params.extend((
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'be',
                                         mm3_col = 1,
                                         mm3_row = i + 1,
                                         mm3_label = cols[0],
                                         value = float(cols[3])),
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'bf',
                                         mm3_col = 2,
                                         mm3_row = i + 1,
                                         mm3_label = cols[0],
                                         value = float(cols[4]))))
                        try:
                            self.params.append(
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'q',
                                         mm3_col = 3,
                                         mm3_row = i + 1,
                                         mm3_label = cols[0],
                                         value = float(cols[5])))
                        # Some bonds parameters don't use bond dipoles.
                        except IndexError:
                            pass
                        continue
                    # Angles.
                    elif match_mm3_angle(line):
                        logger.log(
                            5, '[L{}] Found angle:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [cols[1], cols[2], cols[3]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
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
                        self.params.extend((
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'ae',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = cols[0],
                                     value = float(cols[4])),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'af',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     mm3_label = cols[0],
                                     value = float(cols[5]))))
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
                            5, '[L{}] Found torsion:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [cols[1], cols[2], cols[3], cols[4]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
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
                        self.params.extend((
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = cols[0],
                                     value = float(cols[5])),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     mm3_label = cols[0],
                                     value = float(cols[6])),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     mm3_col = 3,
                                     mm3_row = i + 1,
                                     mm3_label = cols[0],
                                     value = float(cols[7]))))
                        continue
                    # Higher order torsions.
                    # elif match_mm3_higher_torsion(line):
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
                    #     continue
                    # Improper torsions.
                    elif match_mm3_improper(line):
                        logger.log(
                            5, '[L{}] Found torsion:\n{}'.format(
                                i + 1, line.strip('\n')))
                        if section_sub:
                            # Do stuff.
                            atm_lbls = [cols[1], cols[2], cols[3], cols[4]]
                            atm_typs = self.convert_to_types(
                                atm_lbls, self.atom_types[-1])
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
                        self.params.extend((
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'imp1',
                                     mm3_col = 1,
                                     mm3_row = i + 1,
                                     mm3_label = cols[0],
                                     value = float(cols[5])),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'imp2',
                                     mm3_col = 2,
                                     mm3_row = i + 1,
                                     mm3_label = cols[0],
                                     value = float(cols[6]))))
                        continue
                # The Van der Waals are stored in annoying way.
                if line.startswith('-6'):
                    section_vdw = True
                    continue
        logger.log(15, '  -- Read {} parameters.'.format(len(self.params)))
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
            params = self.params
        if lines is None:
            lines = self.lines
        for param in params:
            logger.log(1, '>>> param: {} param.value: {}'.format(
                    param, param.value))
            line = lines[param.mm3_row - 1]
            # There are some problems with this. Probably an optimization
            # technique gave you these crazy parameter values. Ideally, this
            # entire trial FF should be discarded.
            # Someday export_ff should raise an exception when these values
            # get too rediculous, and this exception should be handled by the
            # optimization techniques appropriately.
            if abs(param.value) > 999.:
                logger.warning(
                    'Value of {} is too high! Skipping write.'.format(param))
            elif param.mm3_col == 1:
                lines[param.mm3_row - 1] = (line[:P_1_START] +
                                            '{:10.4f}'.format(param.value) +
                                            line[P_1_END:])
            elif param.mm3_col == 2:
                lines[param.mm3_row - 1] = (line[:P_2_START] +
                                            '{:10.4f}'.format(param.value) +
                                            line[P_2_END:])
            elif param.mm3_col == 3:
                lines[param.mm3_row - 1] = (line[:P_3_START] +
                                            '{:10.4f}'.format(param.value) +
                                            line[P_3_END:])
        with open(path, 'w') as f:
            f.writelines(lines)
        logger.log(10, 'WROTE: {}'.format(path))
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
    return re.match('[\s5a-z][1-5]', mm3_label)
def match_mm3_vdw(mm3_label):
    """Matches MM3* label for bonds."""
    return re.match('[\sa-z]6', mm3_label)
def match_mm3_bond(mm3_label):
    """Matches MM3* label for bonds."""
    return re.match('[\sa-z]1', mm3_label)
def match_mm3_angle(mm3_label):
    """Matches MM3* label for angles."""
    return re.match('[\sa-z]2', mm3_label)
def match_mm3_stretch_bend(mm3_label):
    """Matches MM3* label for stretch-bends."""
    return re.match('[\sa-z]3', mm3_label)
def match_mm3_torsion(mm3_label):
    """Matches MM3* label for all orders of torsional parameters."""
    return re.match('[\sa-z]4|54', mm3_label)
def match_mm3_lower_torsion(mm3_label):
    """Matches MM3* label for torsions (1st through 3rd order)."""
    return re.match('[\sa-z]4', mm3_label)
def match_mm3_higher_torsion(mm3_label):
    """Matches MM3* label for torsions (4th through 6th order)."""
    return re.match('54', mm3_label)
def match_mm3_improper(mm3_label):
    """Matches MM3* label for improper torsions."""
    return re.match('[\sa-z]5', mm3_label)