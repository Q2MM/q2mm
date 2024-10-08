"""
Contains basic data structures used throughout the rest of Q2MM.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import copy
import logging
import numpy as np
import os
import re
import sys

import constants as co
import filetypes

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)

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
# ff_row and ff_col.
class ParamMM3(Param):
    '''
    Adds information to Param that is specific to MM3* parameters.
    '''
    __slots__ = ['atom_labels', 'atom_types', 'ff_col', 'ff_row', 'ff_label']
    def __init__(self, atom_labels=None, atom_types=None, ff_col=None,
                 ff_row=None, ff_label=None,
                 d1=None, d2=None, ptype=None, value=None):
        self.atom_labels = atom_labels
        self.atom_types = atom_types
        self.ff_col = ff_col
        self.ff_row = ff_row
        self.ff_label = ff_label
        super(ParamMM3, self).__init__(ptype=ptype, value=value)
    def __repr__(self):
        return '{}[{}][{},{}]({})'.format(
            self.__class__.__name__, self.ptype, self.ff_row, self.ff_col,
            self.value)
    def __str__(self):
        return '{}[{}][{},{}]({})'.format(
            self.__class__.__name__, self.ptype, self.ff_row, self.ff_col,
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
                                    ff_col = 1,
                                    ff_row = i+1,
                                    value = float(BB[0])),
                            ParamMM3(atom_types = at,
                                    ptype = "be",
                                    ff_col = 2,
                                    ff_row = i+1,
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
                                    ff_col = 1,
                                    ff_row = i + 1,
                                    value = float(BB[0])),
                            ParamMM3(atom_types = at,
                                    ptype = 'ae',
                                    ff_col = 2,
                                    ff_row = i + 1,
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
                                ff_col = 1,
                                ff_row = i + 1,
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
                                ff_col = 1,
                                ff_row = i + 1,
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
                                    ff_col = 1,
                                    ff_row = i + 1,
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
            line = lines[param.ff_row - 1]
            if abs(param.value) > 1999.:
                logger.warning(
                    'Value of {} is too high! Skipping write.'.format(param))
            else:
                atoms = ""
                const = ""
                space3 = " " * 3
                col = int(param.ff_col-1)
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
                    
                lines[param.ff_row - 1] = (atoms+const+'\n')
        with open(path, 'w') as f:
            f.writelines(lines)
        logger.log(10, 'WROTE: {}'.format(path))

class TinkerFF(FF):
    """
    STUFF TO FILL IN LATER
    THE PROBLEM: Depending on the forcefield used, the parameter structures are different.
    mm3.prm (exists)
    amoeba09.prm (development)
    """
    def __init__(self, path=None, data=None, method=None, params=None,
                 score=None):
        super(TinkerFF, self).__init__(path, data, method, params, score)
        self.sub_names = []
        self._atom_types = None
        self._lines = None
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
                    if 'atom' == split[0]:
                        at = split[1]
                        el = split[2]
                        des = split[3][1:-1]
                        atnum = split[4]
                        mass = split[5]
                        #still don't know what this colum does. I don't even
                        # know if its valence
                        # Number of bonds - KJK
                        valence = split[6]
                    if split[0] in bonds:
                        at = [split[1], split[2]]
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                     ptype = 'bf',
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     value = float(split[3])),
                            ParamMM3(atom_types = at,
                                     ptype = 'be',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     value = float(split[4]))))
                    if split[0] in dipoles:
                        at = [split[1], split[2]]
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                     ptype = 'q',
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     value = float(split[3])),
                            #I think this second value is the position of the
                            #dipole along the bond. I've only seen 0.5 which
                            #indicates the dipole is posititioned at the center
                            #of the bond.
                            ParamMM3(atom_types = at,
                                     ptype = 'q_p',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     value = float(split[4]))))
                    if split[0] in pibonds:
                        at = [split[1], split[2]]
                        #I'm still not sure how these effect the potential 
                        # energy but I believe they are correcting factors for
                        # atoms in a pi system with the pi_b being for the bond
                        # and pi_t being for torsions.
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                     ptype = 'pi_b',
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     value = float(split[3])),
                            ParamMM3(atom_types = at,
                                     ptype = 'pi_t',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     value = float(split[4]))))
                    if split[0] in angles:
                        at = [split[1], split[2], split[3]]
                        #TINKER param file might include several equillibrum
                        # bond angles which are for a central atom with 0, 1,
                        # or 2 additional hydrogens on the central atom.
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                    ptype = 'af',
                                    ff_col = 1,
                                    ff_row = i + 1,
                                    value = float(split[4])),
                            ParamMM3(atom_types = at,
                                    ptype = 'ae',
                                    ff_col = 2,
                                    ff_row = i + 1,
                                    value = float(split[5]))))
                        if len(split) == 8:
                            self.params.extend((
                                ParamMM3(atom_types = at,
                                        ptype = 'ae',
                                        ff_col = 3,
                                        ff_row = i + 1,
                                        value = float(split[6])),
                                ParamMM3(atom_types = at,
                                        ptype = 'ae',
                                        ff_col = 4,
                                        ff_row = i + 1,
                                        value = float(split[7]))))
                        elif len(split) == 7:
                            self.params.extend((
                                ParamMM3(atom_types = at,
                                        ptype = 'ae',
                                        ff_col = 3,
                                        ff_row = i + 1,
                                        value = float(split[6]))))
                    if split[0] in torsions:
                        at = [split[1], split[2], split[3], split[4]]
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                    ptype = 'df',
                                    ff_col = 1,
                                    ff_row = i + 1,
                                    value = float(split[5])),
                            ParamMM3(atom_types = at,
                                    ptype = 'df',
                                    ff_col = 2,
                                    ff_row = i + 1,
                                    value = float(split[8])),
                            ParamMM3(atom_types = at,
                                    ptype = 'df',
                                    ff_col = 3,
                                    ff_row = i + 1,
                                    value = float(split[11]))))
                    if 'opbend' == split[0]:
                        at = [split[1], split[2], split[3], split[4]]
                        self.params.append(
                            ParamMM3(atom_types = at,
                                    ptype = 'op_b',
                                    ff_col = 1,
                                    ff_row = i + 1,
                                    value = float(split[5])))
                    if 'vdw' == split[0]:
                    #The first float is the vdw radius, the second has to do
                    # with homoatomic well depths and the last is a reduction
                    # factor for univalent atoms (I don't think we will need
                    # any of these except for the first one).
                        at = [split[1]]
                        self.params.append(
                            ParamMM3(atom_types = at,
                                    ptype = 'vdw',
                                    ff_col = 1,
                                    ff_row = i + 1,
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
            line = lines[param.ff_row - 1]
            if abs(param.value) > 999.:
                logger.warning(
                    'Value of {} is too high! Skipping write.'.format(param))
            else:
                col = int(param.ff_col-1)
                linesplit = line.split()
                value = '{:7.3f}'.format(param.value)
                par = format(linesplit[0],"<10")
                space5 = " "*5
                
                if "pibond" in lines:
                    0
                elif "bond" in line:
                    atoms = "".join([format(el,">5") for el in linesplit[1:3]]) + space5 * 2
                    linesplit[3+col] = value
                    const = "".join([format(el,">12") for el in linesplit[3:]])
                elif "angle" in line:
                    atoms = "".join([format(el,">5") for el in linesplit[1:4]]) + space5  
                    linesplit[4+col] = value
                    const = "".join([format(el,">12") for el in linesplit[4:]])
                elif "torsion" in line:
                    atoms = "".join([format(el,">5") for el in linesplit[1:5]]) + space5  
                    linesplit[5+3*col] = value
                    const = "".join([format(el,">8") for el in linesplit[5:]])
                elif "opbend" in line:
                    atoms = "".join([format(el,">5") for el in linesplit[1:5]]) + space5  
                    linesplit[5+col] = value
                    const = "".join([format(el,">12") for el in linesplit[5:]])
                elif "vdw" in line:
                    atoms = format(split[1]) + space5 * 3
                    linesplit[2+col] = value
                    const = "".join([format(el,">12") for el in linesplit[2:]])
                lines[param.ff_row - 1] = (par+atoms+const+'\n')
        with open(path, 'w') as f:
            f.writelines(lines)
        logger.log(10, 'WROTE: {}'.format(path))


class TinkerMM3A(FF):
    """
    STUFF TO FILL IN LATER
    """
    def __init__(self, path=None, data=None, method=None, params=None,
                 score=None):
        super(TinkerFF, self).__init__(path, data, method, params, score)
        self.sub_names = []
        self._atom_types = None
        self._lines = None
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
                    if 'atom' == split[0]:
                        at = split[1]
                        el = split[2]
                        des = split[3][1:-1]
                        atnum = split[4]
                        mass = split[5]
                        #still don't know what this colum does. I don't even
                        # know if its valence
                        valence = split[6]
                    if split[0] in bonds:
                        at = [split[1], split[2]]
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                     ptype = 'bf',
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     value = float(split[3])),
                            ParamMM3(atom_types = at,
                                     ptype = 'be',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     value = float(split[4]))))
                    if split[0] in dipoles:
                        at = [split[1], split[2]]
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                     ptype = 'q',
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     value = float(split[3])),
                            #I think this second value is the position of the
                            #dipole along the bond. I've only seen 0.5 which
                            #indicates the dipole is posititioned at the center
                            #of the bond.
                            ParamMM3(atom_types = at,
                                     ptype = 'q_p',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     value = float(split[4]))))
                    if split[0] in pibonds:
                        at = [split[1], split[2]]
                        #I'm still not sure how these effect the potential 
                        # energy but I believe they are correcting factors for
                        # atoms in a pi system with the pi_b being for the bond
                        # and pi_t being for torsions.
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                     ptype = 'pi_b',
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     value = float(split[3])),
                            ParamMM3(atom_types = at,
                                     ptype = 'pi_t',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     value = float(split[4]))))
                    if split[0] in angles:
                        at = [split[1], split[2], split[3]]
                        #TINKER param file might include several equillibrum
                        # bond angles which are for a central atom with 0, 1,
                        # or 2 additional hydrogens on the central atom.
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                    ptype = 'af',
                                    ff_col = 1,
                                    ff_row = i + 1,
                                    value = float(split[4])),
                            ParamMM3(atom_types = at,
                                    ptype = 'ae',
                                    ff_col = 2,
                                    ff_row = i + 1,
                                    value = float(split[5]))))
                        if len(split) == 8:
                            self.params.extend((
                                ParamMM3(atom_types = at,
                                        ptype = 'ae',
                                        ff_col = 3,
                                        ff_row = i + 1,
                                        value = float(split[6])),
                                ParamMM3(atom_types = at,
                                        ptype = 'ae',
                                        ff_col = 4,
                                        ff_row = i + 1,
                                        value = float(split[7]))))
                        elif len(split) == 7:
                            self.params.extend((
                                ParamMM3(atom_types = at,
                                        ptype = 'ae',
                                        ff_col = 3,
                                        ff_row = i + 1,
                                        value = float(split[6]))))
                    if split[0] in torsions:
                        at = [split[1], split[2], split[3], split[4]]
                        self.params.extend((
                            ParamMM3(atom_types = at,
                                    ptype = 'df',
                                    ff_col = 1,
                                    ff_row = i + 1,
                                    value = float(split[5])),
                            ParamMM3(atom_types = at,
                                    ptype = 'df',
                                    ff_col = 2,
                                    ff_row = i + 1,
                                    value = float(split[8])),
                            ParamMM3(atom_types = at,
                                    ptype = 'df',
                                    ff_col = 3,
                                    ff_row = i + 1,
                                    value = float(split[11]))))
                    if 'opbend' == split[0]:
                        at = [split[1], split[2], split[3], split[4]]
                        self.params.append(
                            ParamMM3(atom_types = at,
                                    ptype = 'op_b',
                                    ff_col = 1,
                                    ff_row = i + 1,
                                    value = float(split[5])))
                    if 'vdw' == split[0]:
                    #The first float is the vdw radius, the second has to do
                    # with homoatomic well depths and the last is a reduction
                    # factor for univalent atoms (I don't think we will need
                    # any of these except for the first one).
                        at = [split[1]]
                        self.params.append(
                            ParamMM3(atom_types = at,
                                    ptype = 'vdw',
                                    ff_col = 1,
                                    ff_row = i + 1,
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
            line = lines[param.ff_row - 1]
            if abs(param.value) > 999.:
                logger.warning(
                    'Value of {} is too high! Skipping write.'.format(param))
            #Currently this isn't to flexible. The prm file (or atleast the 
            # parts that are actually being paramterized have to be formatted
            # correctly. This includes the position of the columns and a space
            # at the end of every line.
            else:
                col = int(param.ff_col-1)
                pos = 12*(col + 1)
                linesplit = line.split()
                value = '{:7.4f}'.format(param.value)
                par = " "*12 # (12 * 1)
                n = len(linesplit[0])
                par[:n] = linesplit[0]    
                atoms = " "*5*4 # (5 * 4)
                const = " "*4*12 # (4 * 12)
                
                if "pibond" in lines:
                    0
                # bond A B Kb b (3+(n-1)) 
                elif "bond" in line:
                    n1 = len(linesplit[1])
                    n2 = len(linesplit[2])
                    atoms[4-n1:4] = linesplit[1]
                    atoms[8-n2:8] = linesplit[2]
                    n3 = len(value)
                    const[pos-n3:pos] = value
#                    linesplit[3+col] = value
                # angle A B C (4+(n-1))
                elif "angle" in line:
                    n1 = len(linesplit[1])
                    n2 = len(linesplit[2])
                    n3 = len(linesplit[3])
                    atoms[4-n1:4] = linesplit[1]
                    atoms[8-n2:8] = linesplit[2]
                    atoms[12-n3:12] = linesplit[3]
                    n4 = len(value)
                    const[pos-n4:pos] = value
                    #linesplit[4+col] = value
                # torsion A B C D (5+3*(n-1))
                elif "torsion" in line:
                    linesplit[5+3*col] = value
                # opbend A B C D (5)
                elif "opbend" in line:
                    n1 = len(linesplit[1])
                    n2 = len(linesplit[2])
                    n3 = len(linesplit[3])
                    n4 = len(linesplit[4])
                    atoms[4-n1:4] = linesplit[1]
                    atoms[8-n2:8] = linesplit[2]
                    atoms[12-n3:12] = linesplit[3]
                    atoms[16-n4:16] = linesplit[4]
                    n5 = len(value)
                    const[pos-n5:pos] = value
#                    linesplit[5+col] = value
#                lines[param.ff_row - 1] = ("\t".join(linesplit)+"\n")
                lines[param.ff_row - 1] = (par+atoms+const+'\n')
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
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     value = float(rad)),
                            ParamMM3(atom_types = atm,
                                     ptype = 'vdwe',
                                     ff_col = 2,
                                     ff_row = i + 1,
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
                                         ff_col = 1,
                                         ff_row = i + 1,
                                         ff_label = line[:2],
                                         value = parm_cols[0]),
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'bf',
                                         ff_col = 2,
                                         ff_row = i + 1,
                                         ff_label = line[:2],
                                         value = parm_cols[1])))
                        try:
                            self.params.append(
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'q',
                                         ff_col = 3,
                                         ff_row = i + 1,
                                         ff_label = line[:2],
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
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     ff_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'af',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     ff_label = line[:2],
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
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     ff_label = line[:2],
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
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     ff_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     ff_label = line[:2],
                                     value = parm_cols[1]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     ff_col = 3,
                                     ff_row = i + 1,
                                     ff_label = line[:2],
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
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     ff_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     ff_label = line[:2],
                                     value = parm_cols[1]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     ff_col = 3,
                                     ff_row = i + 1,
                                     ff_label = line[:2],
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
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     ff_label = line[:2],
                                     value = parm_cols[0]),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'imp2',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     ff_label = line[:2],
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
                                         ff_col = 1,
                                         ff_row = i + 1,
                                         ff_label = line[:2],
                                         value = parm_cols[0]),
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'vdwfc',
                                         ff_col = 2,
                                         ff_row = i + 1,
                                         ff_label = line[:2],
                                         value = parm_cols[1])))
                        continue
                # The Van der Waals are stored in annoying way.
                if line.startswith('-6'):
                    section_vdw = True
                    continue
                if section_vdw and line.startswith('-2'):
                    section_vdw = False
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
                #                      ff_col = 1,
                #                      ff_row = i + 1,
                #                      value = float(rad)),
                #             ParamMM3(atom_types = atm,
                #                      ptype = 'vdwe',
                #                      ff_col = 2,
                #                      ff_row = i + 1,
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
                                         ff_col = 1,
                                         ff_row = i + 1,
                                         ff_label = cols[0],
                                         value = float(cols[3])),
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'bf',
                                         ff_col = 2,
                                         ff_row = i + 1,
                                         ff_label = cols[0],
                                         value = float(cols[4]))))
                        try:
                            self.params.append(
                                ParamMM3(atom_labels = atm_lbls,
                                         atom_types = atm_typs,
                                         ptype = 'q',
                                         ff_col = 3,
                                         ff_row = i + 1,
                                         ff_label = cols[0],
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
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     ff_label = cols[0],
                                     value = float(cols[4])),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'af',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     ff_label = cols[0],
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
                    #                  ff_col = 1,
                    #                  ff_row = i + 1,
                    #                  ff_label = line[:2],
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
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     ff_label = cols[0],
                                     value = float(cols[5])),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     ff_label = cols[0],
                                     value = float(cols[6])),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'df',
                                     ff_col = 3,
                                     ff_row = i + 1,
                                     ff_label = cols[0],
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
                    #                  ff_col = 1,
                    #                  ff_row = i + 1,
                    #                  ff_label = cols[0],
                    #                  value = parm_cols[0]),
                    #         ParamMM3(atom_labels = atm_lbls,
                    #                  atom_types = atm_typs,
                    #                  ptype = 'df',
                    #                  ff_col = 2,
                    #                  ff_row = i + 1,
                    #                  ff_label = cols[0],
                    #                  value = parm_cols[1]),
                    #         ParamMM3(atom_labels = atm_lbls,
                    #                  atom_types = atm_typs,
                    #                  ptype = 'df',
                    #                  ff_col = 3,
                    #                  ff_row = i + 1,
                    #                  ff_label = cols[0],
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
                                     ff_col = 1,
                                     ff_row = i + 1,
                                     ff_label = cols[0],
                                     value = float(cols[5])),
                            ParamMM3(atom_labels = atm_lbls,
                                     atom_types = atm_typs,
                                     ptype = 'imp2',
                                     ff_col = 2,
                                     ff_row = i + 1,
                                     ff_label = cols[0],
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
            line = lines[param.ff_row - 1]
            # There are some problems with this. Probably an optimization
            # technique gave you these crazy parameter values. Ideally, this
            # entire trial FF should be discarded.
            # Someday export_ff should raise an exception when these values
            # get too rediculous, and this exception should be handled by the
            # optimization techniques appropriately.
            if abs(param.value) > 999.:
                logger.warning(
                    'Value of {} is too high! Skipping write.'.format(param))
            elif param.ff_col == 1:
                lines[param.ff_row - 1] = (line[:P_1_START] +
                                            '{:10.4f}'.format(param.value) +
                                            line[P_1_END:])
            elif param.ff_col == 2:
                lines[param.ff_row - 1] = (line[:P_2_START] +
                                            '{:10.4f}'.format(param.value) +
                                            line[P_2_END:])
            elif param.ff_col == 3:
                lines[param.ff_row - 1] = (line[:P_3_START] +
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

def match_mm3_label(ff_label):
    """
    Makes sure the MM3* label is recognized.

    The label is the 1st 2 characters in the line containing the parameter
    in a Schrodinger mm3.fld file.
    """
    return re.match('[\s5a-z][1-5]', ff_label)
def match_mm3_vdw(ff_label):
    """Matches MM3* label for bonds."""
    return re.match('[\sa-z]6', ff_label)
def match_mm3_bond(ff_label):
    """Matches MM3* label for bonds."""
    return re.match('[\sa-z]1', ff_label)
def match_mm3_angle(ff_label):
    """Matches MM3* label for angles."""
    return re.match('[\sa-z]2', ff_label)
def match_mm3_stretch_bend(ff_label):
    """Matches MM3* label for stretch-bends."""
    return re.match('[\sa-z]3', ff_label)
def match_mm3_torsion(ff_label):
    """Matches MM3* label for all orders of torsional parameters."""
    return re.match('[\sa-z]4|54', ff_label)
def match_mm3_lower_torsion(ff_label):
    """Matches MM3* label for torsions (1st through 3rd order)."""
    return re.match('[\sa-z]4', ff_label)
def match_mm3_higher_torsion(ff_label):
    """Matches MM3* label for torsions (4th through 6th order)."""
    return re.match('54', ff_label)
def match_mm3_improper(ff_label):
    """Matches MM3* label for improper torsions."""
    return re.match('[\sa-z]5', ff_label)

def mass_weight_hessian(hess, atoms, reverse=False):
    """
    Mass weights Hessian. If reverse is True, it un-mass weights
    the Hessian.
    """
    masses = [co.MASSES[x.element] for x in atoms if not x.is_dummy]
    changes = []
    for mass in masses:
        changes.extend([1 / np.sqrt(mass)] * 3)
    x, y = hess.shape
    for i in range(0, x):
        for j in range(0, y):
            if reverse:
                hess[i, j] = \
                    hess[i, j] / changes[i] / changes[j]
            else:
                hess[i, j] = \
                    hess[i, j] * changes[i] * changes[j]

def mass_weight_eigenvectors(evecs, atoms, reverse=False):
    """
    Mass weights eigenvectors. If reverse is True, it un-mass weights
    the eigenvectors.
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

def replace_minimum(array, value=1):
    """
    Replace the minimum vallue in an arbitrary NumPy array. Historically,
    the replace value is either 1 or co.HESSIAN_CONVERSION.
    """
    minimum = array.min()
    minimum_index = np.where(array == minimum)
    assert minimum < 0, 'Minimum of array is not negative!'
    # It would be better to address this in a different way. This particular
    # data structure just isn't what we want.
    array.setflags(write=True)
    # Sometimes we use 1, but sometimes we use co.HESSIAN_CONVERSION.
    array[minimum_index] = value
    logger.log(1, '>>> minimum_index: {}'.format(minimum_index))
    logger.log(1, '>>> array:\n{}'.format(array))
    logger.log(10, '  -- Replaced minimum in array with {}.'.format(value))

def check_mm_dummy(hess, dummy_indices):
    """
    Removes dummy atom rows and columns from the Hessian based upon
    dummy_indices.

    Arguments
    ---------
    hess : np.matrix
    dummy_indices : list of integers
                    Integers correspond to the indices to be removed from the
                    np.matrix of the Hessian.

    Returns
    -------
    np.matrix
    """
    hess = np.delete(hess, dummy_indices, 0)
    hess = np.delete(hess, dummy_indices, 1)
    logger.log(15, 'Created {} Hessian w/o dummy atoms.'.format(hess.shape))
    return hess

def get_dummy_hessian_indices(dummy_indices):
    """
    Takes a list of indices for the dummy atoms and returns another list of
    integers corresponding to the rows of the eigenvectors to remove
    for those those dummy atoms.

    Arguments
    ---------
    dummy_indices : list of integers
                    Indices for the dummy atoms.

    Returns
    -------
    list of integers
    """
    hess_dummy_indices = []
    for index in dummy_indices:
        hess_index = (index - 1) * 3
        hess_dummy_indices.append(hess_index)
        hess_dummy_indices.append(hess_index + 1)
        hess_dummy_indices.append(hess_index + 2)
    return hess_dummy_indices
