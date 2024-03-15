#!/usr/bin/env python
"""
Handles importing data from the various filetypes that Q2MM uses.

Schrodinger
-----------
When importing Schrodinger files, if the atom.typ file isn't in the directory
where you execute the Q2MM Python scripts, you may see this warning:

  WARNING mmat_get_atomic_num x is not a valid atom type
  WARNING mmat_get_mmod_name x is not a valid atom type

In this example, x is the number of a custom atom type defined and added to
atom.typ. The warning can be ignored. If it's bothersome, copy atom.typ into
the directory where you execute the Q2MM Python scripts.

Note that the atom.typ must be located with your structure files, else the
Schrodinger jobs will fail.
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

try:
    from schrodinger import structure as sch_str
    from schrodinger.application.jaguar import input as jag_in
except:
    print("Schrodinger not installed, limited functionality")
    pass

import constants as co
import datatypes

logging.config.dictConfig(co.LOG_SETTINGS)
logger = logging.getLogger(__file__)


def check_licenses():
    max_fails=5
    max_timeout=None
    timeout=10
    check_tokens=True
    current_directory = os.getcwd()
    #os.chdir(directory)
    current_timeout = 0
    current_fails = 0
    licenses_available = False

    logger.log(5, "  -- Checking Schrodinger tokens.")
    while True:
        token_string = sp.check_output(
            '$SCHRODINGER/utilities/licutil -available', shell=True)
        if (sys.version_info > (3, 0)):
          token_string = token_string.decode("utf-8")
        suite_tokens = co.LIC_SUITE.search(token_string)
        macro_tokens = co.LIC_MACRO.search(token_string)
        #suite_tokens = re.search(co.LIC_SUITE, token_string)
        #macro_tokens = re.search(co.LIC_MACRO, token_string)
        if not suite_tokens and not macro_tokens:
            raise Exception(
                'The command "$SCHRODINGER/utilities/licutil '
                '-available" is not working with the current '
                'regex in calculate.py.\nOUTPUT:\n{}'.format(
                    token_string))
        suite_tokens = int(suite_tokens.group(1))
        macro_tokens = int(macro_tokens.group(1))
        if suite_tokens <= co.MIN_SUITE_TOKENS and \
                macro_tokens <= co.MIN_MACRO_TOKENS:
            if max_timeout is not None and \
                    current_timeout > max_timeout:
                
                raise Exception(
                    "Not enough tokens to run. Waited {} seconds "
                    "before giving up.".format(
                        current_timeout))
            current_timeout += timeout
            time.sleep(timeout)
        else:
            break

    
    return macro_tokens, suite_tokens

        




def pretty_timeout(current_timeout, macro_tokens, suite_tokens, end=False,
                   level=10, name_com=None):
    """
    Logs information about the wait for Schrodinger tokens.

    Arguments
    ---------
    current_timeout : int
                      Number of times waited for Schrodinger tokens.
    macro_tokens : int
                   Current number of available MacroModel tokens.
    suite_tokens : int
                   Current number of available Schrodinger Suite tokens.
    end : bool
          If True, adds a pretty ending border to all these logs.
    level : int
            Logging level of the pretty messages.
    """
    if current_timeout == 0:
        if name_com:
            logger.warning('  -- Waiting on tokens to run {}.'.format(
                    name_com))
        logger.log(level,
                   '--' + ' (s) '.center(8, '-') +
                   '--' + ' {} '.format(co.LABEL_SUITE).center(17, '-') +
                   '--' + ' {} '.format(co.LABEL_MACRO).center(17, '-') +
                   '--')
    logger.log(level, '  {:^8d}  {:^17d}  {:^17d}'.format(
            current_timeout, macro_tokens, suite_tokens))
    if end is True:
        logger.log(level, '-' * 50)