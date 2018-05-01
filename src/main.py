#!/usr/bin/env python3

import argparse
import os

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

    parser.add_argument('-dataPath', type=str,
                        help='Path to save fetched data.')
    parser.add_argument('-fetch', type=bool,
                        help='If true, {dataPath} will be deleted and the data will be fetched again')

    args = parser.parse_args()

    if args.dataPath == None:
        raise AttributeError("Specify correct path for data")

    if os.path.exists(args.dataPath):
        if not os.path.isdir(args.dataPath):
            raise AttributeError("Specify path doesn't lead to directory")

    if args.fetch == None:
        args.fetch = False

    return args


def main():
    argv = parseArgv()


if __name__ == "__main__":
    main()