"""
SCANNER module provides functions for accessing files in vendor-specific formats.
"""
from __future__ import (absolute_import, division, print_function, unicode_literals)

__author__ = 'vlad'
__version__ = 0.1

import numpy as np

try:
    import openslide
except:
    raise ImportError('This module needs OPENSLIDE library and its Python API')
