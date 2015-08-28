# -*- coding: utf-8 -*-
"""
WSI_CODEBOOK_INFO

Prints information about a codebook.

@author: vlad
"""

from __future__ import (absolute_import, division, print_function, unicode_literals)
from builtins import *

__author__ = 'Vlad Popovici'
__version__ = 0.01

import argparse as opt

from util.storage import ModelPersistence

def main():
    p = opt.ArgumentParser(description="""
    Prints information about a codebook.
    """)
    p.add_argument('cbk', action='store', help='codebook file name')
    p.add_argument('-s', '--size', action='store_true', help='print codebook size', default=True)
    p.add_argument('-n', '--norm', action='store_true', help='print 1 if data was normalized', default=False)

    args = p.parse_args()
    cbk_file = args.cbk

    with ModelPersistence(cbk_file, 'r', format='pickle') as mp:
        codebook = mp['codebook']
        standardize = mp['standardize']
    
        if args.size:
            print(codebook.cluster_centers_.shape[0])
        if args.norm:
            print("1" if standardize else "0")

    return


if __name__ == '__main__':
    main()

