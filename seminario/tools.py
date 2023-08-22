"""
This module contain a function to run the Seminario tool
"""
from __future__ import division, print_function, absolute_import

def seminario_ff() :
    """
    The entry point for the Seminario script.

    Setting up argument parser, load files and then estimate force field.
    """

    import argparse
    import os
    
    import numpy as np
    import parmed

    import gaussian
    import forcefield

    argparser = argparse.ArgumentParser(description="Script to compute bond force constant with Seminario")
    argparser.add_argument('-f', '--checkpoint', help="the formated checkpoint file")
    argparser.add_argument('-s', '--struct', help="a structure file")
    argparser.add_argument('-b', '--bonds', nargs="+", help="the bonds")
    argparser.add_argument('-a', '--angles', nargs="+", help="the angles")
    argparser.add_argument('--scaling', type=float, help="the frequency scaling factor", default=0.963)
    argparser.add_argument('--saveopt', action='store_true', help="save the optimized coordinates to file", default=False)
    args = argparser.parse_args()

    if args.checkpoint is None or args.struct is None :
        print("Nothing to be done. Use -h to see help. \n Exit.")
        return

    struct = parmed.load_file(args.struct, skip_bonds=True)
    xyz_opt, hess = gaussian.parse_fchk(args.checkpoint)
    xyz_orig = np.array(struct.coordinates)
    struct.coordinates = xyz_opt*0.529177249 # 0.529177249 is Bohr to A

    if args.saveopt :
        base, ext = os.path.splitext(args.struct)
        struct.save(base+"_opt"+ext)

    if args.bonds is not None:
        forcefield.make_bonded_ff(struct, xyz_orig, hess,
                                    args.bonds, args.scaling)

    if args.angles is not None:
        forcefield.make_angled_ff(struct, xyz_orig, hess,
                                    args.angles, args.scaling)

seminario_ff()
