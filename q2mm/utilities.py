#!/usr/bin/env python
"""
Contains basic utility methods for use in Q2MM.
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from argparse import RawTextHelpFormatter
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

def convert_atom_type(atom_type:str) -> str:
    q2mm_atom_type = atom_type
    if '.' in atom_type:
        q2mm_atom_type.replace(".", "")
    if len(q2mm_atom_type) > 1 and q2mm_atom_type[1].isupper():
        q2mm_atom_type[1] = q2mm_atom_type[1].lower()
    return q2mm_atom_type

def convert_atom_type_pair(atom_type_pair):
    q2mm_atom_type_pair = [convert_atom_type(atom_type) for atom_type in atom_type_pair]
    return q2mm_atom_type_pair

def convert_atom_types(atom_type_pairs:list) -> list:
    q2mm_atom_type_pairs = [convert_atom_type_pair(atom_type_pair) for atom_type_pair in atom_type_pairs]
    return q2mm_atom_type_pairs