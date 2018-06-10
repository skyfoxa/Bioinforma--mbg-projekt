#!/usr/bin/env python3

from .iStatistics import iStatistics
from src.Statistics.StatisticsUtilities import *
import numpy as np

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

class ChiSquared(iStatistics):

    threshold01 = 6.635
    c1 = 0
    c2 = 0

    def __init__(self, geneMatrix1, geneMatrix2):
        super().__init__(geneMatrix1, geneMatrix2)

    def compute(self):
        for colM1 in self.geneMatrix1.T:
            for colM2 in self.geneMatrix2.T:
                self.__computeCols__(colM1, colM2)

        print(self.c1)
        print(self.c2)


    def __computeCols__(self, colM1, colM2):
        expected = StatisticsUtilities.calculateExpected(colM1, colM2)
        observed = StatisticsUtilities.calculateObserved(colM1, colM2)

        pVal = np.sum(np.divide(np.square(np.subtract(observed, expected)), expected))

        if pVal > self.threshold01:
            # print(pVal)
            self.c1 += 1
        else:
            self.c2 += 1


    def validate(self):
        raise Exception("iStatistics - validate(self) not implemented")

    def plot(self):
        raise Exception("iStatistics - plot(self) not implemented")

    def testExpected(self):
        col1 = np.array([1, 0, 1, 0, 0])
        col2 = np.array([1, 0, 0, 1, 0])

        result = StatisticsUtilities.calculateExpected(col1, col2)

        realResult = np.array([[0.8, 1.2], [1.2, 1.8]], dtype=float)
        np.testing.assert_allclose(result, realResult, rtol=1e-6)

    def testObserved(self):
        col1 = np.array([1, 0, 1, 0, 0])
        col2 = np.array([1, 0, 0, 1, 0])

        result = StatisticsUtilities.calculateObserved(col1, col2)

        realResult = np.array([[1, 1], [1, 2]], dtype=float)
        np.testing.assert_array_equal(result, realResult)

    def testComputeCols(self):
        self.testExpected()
        self.testObserved()

        col1 = np.array([1, 0, 1, 0, 0])
        col2 = np.array([1, 0, 0, 1, 0])
        self.__computeCols__(col1, col2)