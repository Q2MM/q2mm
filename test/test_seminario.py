
from __future__ import division, print_function, absolute_import

import os
import sys
import time
import unittest

import seminario
import parmed
import numpy as np

fchk_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..","examples","model.fchk")
gro_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..","examples","model.gro")
opt_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..","examples","model_opt.gro")

class PutOutput(object) :

    def __init__(self) :
        self.out = []

    def write(self, str) :
        self.out.append(str)

    def __str__(self) :
        return "".join(self.out)

class TestSeminario(unittest.TestCase):

    def test_loadstruct(self) :

        struct = parmed.load_file(gro_filename, skip_bonds=True)
        self.assertEqual(len(struct.atoms),41)

    def test_loadfchk(self) :

        xyz_opt, hess = seminario.gaussian.parse_fchk(fchk_filename)
        self.assertEqual(xyz_opt.shape,(41,3))
        self.assertEqual(hess.shape,(123,123))
        self.assertEqual(xyz_opt[0,0],-2.39626372)
        self.assertEqual(hess[0,0],0.493623378)

    def test_bondparm(self) :

        struct = parmed.load_file(gro_filename, skip_bonds=True)
        xyz_opt, hess = seminario.gaussian.parse_fchk(fchk_filename)
        struct.coordinates = xyz_opt*0.529177249
        atom1 = struct.view[":1@N"].atoms[0]
        atom2 = struct.view[":3@CU"].atoms[0]
        bond = parmed.Bond(atom1, atom2)
        force = seminario.forcefield.seminario_bond(bond, hess, 0.963)
        self.assertEqual(np.round(force,4), 60493.9023)
        self.assertEqual(np.round(bond.measure(),3), 1.993)

    def test_angleparm(self) :

        struct = parmed.load_file(gro_filename, skip_bonds=True)
        xyz_opt, hess = seminario.gaussian.parse_fchk(fchk_filename)
        struct.coordinates = xyz_opt*0.529177249
        atom1 = struct.view[":1@CG"].atoms[0]
        atom2 = struct.view[":1@ND1"].atoms[0]
        atom3 = struct.view[":3@CU"].atoms[0]
        angle = parmed.Angle(atom1, atom2, atom3)
        force = seminario.forcefield.seminario_angle(angle, hess, 0.963)
        self.assertEqual(np.round(force,4), 192.3902)
        self.assertEqual(np.round(angle.measure(),3), 123.642)

    def test_makebonds(self) :

        stdout_orig = sys.stdout
        sys.stdout = PutOutput()

        struct = parmed.load_file(gro_filename, skip_bonds=True)
        xyz_opt, hess = seminario.gaussian.parse_fchk(fchk_filename)
        xyz_orig = np.array(struct.coordinates)
        struct.coordinates = xyz_opt*0.529177249
        seminario.forcefield.make_bonded_ff(struct, xyz_orig, hess,
                                                [":1@N-:3@CU"], 0.963)
        seminario.forcefield.make_bonded_ff(struct, xyz_orig, hess,
                                                [":1@N-:3@CU"], 1.0)

        expected_out = """Bond	idx1	idx2	r(x-ray)	r(opt) [nm]	k [kJ/mol/nm2]
:1@N-:3@CU	3	38	0.2185	0.1993	60493.9023
Bond	idx1	idx2	r(x-ray)	r(opt) [nm]	k [kJ/mol/nm2]
:1@N-:3@CU	3	38	0.2185	0.1993	62818.1747\n"""
        actual_out = str(sys.stdout)
        sys.stdout = stdout_orig

        self.assertEqual(expected_out, actual_out)

    def test_makeangles(self) :

        stdout_orig = sys.stdout
        sys.stdout = PutOutput()

        struct = parmed.load_file(gro_filename, skip_bonds=True)
        xyz_opt, hess = seminario.gaussian.parse_fchk(fchk_filename)
        xyz_orig = np.array(struct.coordinates)
        struct.coordinates = xyz_opt*0.529177249
        seminario.forcefield.make_angled_ff(struct, xyz_orig, hess,
                                                [":1@CG-:1@ND1-:3@CU"], 0.963)
        seminario.forcefield.make_angled_ff(struct, xyz_orig, hess,
                                                [":1@CG-:1@ND1-:3@CU"], 1.0)

        expected_out = """Angle	idx1	idx2	idx3	theta(x-ray)	theta(opt)	k [kJ/mol]
:1@CG-:1@ND1-:3@CU	9	10	38	126.0279	123.6422	192.3902
Angle	idx1	idx2	idx3	theta(x-ray)	theta(opt)	k [kJ/mol]
:1@CG-:1@ND1-:3@CU	9	10	38	126.0279	123.6422	199.7822\n"""
        actual_out = str(sys.stdout)
        sys.stdout = stdout_orig

        self.assertEqual(expected_out, actual_out)

    def test_toolnothing(self) :

        stdout_orig = sys.stdout
        sys.stdout = PutOutput()

        sys.argv = ["seminario_ff"]
        seminario.tools.seminario_ff()
        expected_out = "Nothing to be done. Use -h to see help."
        actual_out = str(sys.stdout)
        sys.stdout = stdout_orig

        self.assertEqual(expected_out, actual_out.split("\n")[0].strip())

    def test_toolnormal(self) :

        stdout_orig = sys.stdout
        sys.stdout = PutOutput()

        sys.argv = ["seminario_ff", "-f", fchk_filename, "-s",
                    gro_filename, "-a",
                    ":1@CG-:1@ND1-:3@CU", "-b", ":1@N-:3@CU","--saveopt"]
        seminario.tools.seminario_ff()
        expected_out = """Bond	idx1	idx2	r(x-ray)	r(opt) [nm]	k [kJ/mol/nm2]
:1@N-:3@CU	3	38	0.2185	0.1993	60493.9023
Angle	idx1	idx2	idx3	theta(x-ray)	theta(opt)	k [kJ/mol]
:1@CG-:1@ND1-:3@CU	9	10	38	126.0279	123.6422	192.3902\n"""
        actual_out = str(sys.stdout)
        sys.stdout = stdout_orig

        self.assertEqual(expected_out, actual_out)

        for i in range(10) :
            time.sleep(1)
            if os.path.exists(opt_filename) :
                break
        self.assertTrue(os.path.exists(opt_filename))
        os.remove(opt_filename)


if __name__ == '__main__':

    unittest.main()
