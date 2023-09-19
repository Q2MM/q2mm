#!/usr/bin/env python
"""
Contains basic utility methods for use in Q2MM.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from argparse import RawTextHelpFormatter
from collections import Counter
from string import digits
import logging
from logging import config
import mmap
import numpy as np
import math
import os
import re
import subprocess as sp
import time
import sys

# region Atom Type Conversion


def convert_atom_type(atom_type: str) -> str:
    """_summary_

    Args:
        atom_type (str): _description_

    Returns:
        str: _description_
    """
    q2mm_atom_type = ''.join(filter(str.isalnum, atom_type))
    q2mm_atom_type = q2mm_atom_type.upper()
    # TODO: MF Add a check to verify it is included in atom.typ here, 
    # exception should be caught, propagated, and handled here to avoid 
    # silent failure within MacroModel upon FF export (or other silent or loud failures).
    return q2mm_atom_type


def convert_atom_type_pair(atom_type_pair):
    q2mm_atom_type_pair = [convert_atom_type(atom_type) for atom_type in atom_type_pair]
    return q2mm_atom_type_pair


def convert_atom_types(atom_type_pairs: list) -> list:
    q2mm_atom_type_pairs = [
        convert_atom_type_pair(atom_type_pair) for atom_type_pair in atom_type_pairs
    ]
    return q2mm_atom_type_pairs

def is_same_bond(atom_type_pair1:list, atom_type_pair2:list) -> bool:
    if atom_type_pair1[0] == atom_type_pair2[0] and atom_type_pair1[1] == atom_type_pair2[1]:
        return True
    elif atom_type_pair1[1] == atom_type_pair2[0] and atom_type_pair1[0] == atom_type_pair2[1]:
        return True
    else:
        return False

# endregion Atom Type Conversion
