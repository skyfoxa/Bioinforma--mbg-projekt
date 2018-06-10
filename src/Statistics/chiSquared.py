#!/usr/bin/env python3

from .iStatistics import iStatistics
from src.Statistics.StatisticsUtilities import *
import numpy as np

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

class ChiSquared(iStatistics):
    def __init__(self, geneMatrix1, geneMatrix2):
        super().__init__(geneMatrix1, geneMatrix2)

    def compute(self):
        for colM1 in self.geneMatrix1.T:
            for colM2 in self.geneMatrix2.T:
                self.__computeCols__(colM1, colM2)


    def __computeCols__(self, colM1, colM2):
        StatisticsUtilities.calculateExpected(colM1, colM2)
        StatisticsUtilities.calculateObserved(colM1, colM2)
        ...

    def validate(self):
        raise Exception("iStatistics - validate(self) not implemented")

    def plot(self):
        raise Exception("iStatistics - plot(self) not implemented")

    def test(self):
        col1 = np.array([1, 0, 1, 0, 0])
        col2 = np.array([1, 0, 0, 0, 0])

        result = StatisticsUtilities.calculateExpected(col1, col2)

        realResult = np.array([[0.4, 1.6], [0.6, 2.4]], dtype=float)
        np.testing.assert_allclose(result, realResult, rtol=1e-6)