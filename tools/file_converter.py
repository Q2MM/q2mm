from openbabel import openbabel as ob
from openbabel import pybel as pb
import numpy as np
import glob, os
import multiprocessing as mp
from multiprocessing import Pool


def mae_to_mol2(infile,outfile):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("mae","mol2")
    mol = ob.OBMol()
    conv.ReadFile(mol, infile)
    conv.WriteFile(mol, outfile)

def mol2_to_mae(infile,outfile):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("mol2","mae")
    mol = ob.OBMol()
    conv.ReadFile(mol, infile)
    conv.WriteFile(mol, outfile)

def fchk_to_mol2(infile,outfile):
    conv = ob.OBConversion()
    conv.SetInAndOutFormats("fchk","mol2")
    mol = ob.OBMol()
    conv.ReadFile(mol, infile)
    conv.WriteFile(mol, outfile)