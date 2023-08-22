"""
The seminario package provides tools to estimate force field parameters
using the method of Seminario
"""

__author__ = 'Samuel Genheden'
__all__ = ['gaussian', 'forcefield', 'tools']

from seminario import gaussian, forcefield, tools

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
