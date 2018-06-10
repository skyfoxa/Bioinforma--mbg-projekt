#!/usr/bin/env python3

from .iStatistics import iStatistics

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

class ChiSquared(iStatistics):
    def __init__(self, geneMatrix1, geneMatrix2):
        super().__init__(geneMatrix1, geneMatrix2)

    def compute(self):
        raise Exception("iStatistics - compute(self) not implemented")

    def computeCol(self):
        ...

    def validate(self):
        raise Exception("iStatistics - validate(self) not implemented")

    def plot(self):
        raise Exception("iStatistics - plot(self) not implemented")