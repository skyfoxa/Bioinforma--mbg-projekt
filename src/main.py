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

def tests():
    ChiSquared(None, None).testExpected()
    ChiSquared(None, None).testObserved()


def main():
    argv = parseArgv()
    dataHandler = DataHandler(gene1=argv.gene1, gene2=argv.gene2)
    gene1Filtered = BooleanReferenceSeqFilter(dataHandler.gene1Data, dataHandler.referenceGene1).filterData()
    gene2Filtered = BooleanReferenceSeqFilter(dataHandler.gene2Data, dataHandler.referenceGene2).filterData()

    chiSquared = ChiSquared(gene1Filtered, gene2Filtered)
    chiSquared.compute()

if __name__ == "__main__":
    tests()
    # main()