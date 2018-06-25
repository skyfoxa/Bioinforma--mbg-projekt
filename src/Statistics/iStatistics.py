#!/usr/bin/env python3

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

class iStatistics(object):
    geneMatrix1 = None
    geneMatrix2 = None

    def __init__(self, geneMatrix1, geneMatrix2):
        self.geneMatrix1 = geneMatrix1
        self.geneMatrix2 = geneMatrix2

    def compute(self):
        raise Exception("iStatistics - compute(self) not implemented")

    def validate(self):
        raise Exception("iStatistics - validate(self) not implemented")

    def getResults(self):
        raise Exception("iStatistics - getResults(self) not implemented")

    def printResult(self):
        raise Exception("iStatistics - printResult(self) not implemented")

