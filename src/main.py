#!/usr/bin/env python3

from src import DataHandler

__authors__ = "Marek Zvara, Marek Hrvol"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz"
__description__ = "MBG"

def main():
    pass

if __name__ == "__main__":
    dataHandler = DataHandler(shouldFetch=True)
    dataHandler.fetchAllDataIfNeededFor(genes=[])