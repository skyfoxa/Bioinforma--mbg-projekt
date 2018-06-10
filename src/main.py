#!/usr/bin/env python3

import argparse
import os
from src import DataHandler
from src.Statistics import *
from src.models.Filters import *

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

def parseArgv():
    """Parse input parameters with argparse.

        Args:

        Returns:
            The return value. Returns object with input params:

    """
    parser = argparse.ArgumentParser(description='Gen correlation detector')

    parser.add_argument('-gene1', type=str,
                        help='Path to gene 1 - .ped file.')

    parser.add_argument('-gene2', type=str,
                        help='Path to gene 2 - .ped file.')

    args = parser.parse_args()

    if args.gene1 == None or args.gene2 == None:
        raise AttributeError("Specify correct path for genes data")

    if not os.path.exists(args.gene1):
        raise AttributeError("Specify path doesn't lead to .ped file")

    if not os.path.exists(args.gene2):
        raise AttributeError("Specify path doesn't lead to .ped file")

    return args


def main():
    # argv = parseArgv()
    # dataHandler = DataHandler(gene1=argv.gene1, gene2=argv.gene2)
    chi = ChiSquared(None, None)
    chi.test()

if __name__ == "__main__":
    main()