#!/usr/bin/env python3

from src.Statistics.statisticsUtilities import *
from .iStatistics import iStatistics
import src.models.statisticsCorrelation as sc

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"


class ChiSquared(iStatistics):
    correlations = []

    def __init__(self, geneMatrix1, geneMatrix2):
        super().__init__(geneMatrix1, geneMatrix2)

    def compute(self):
        self.correlations = []
        for idx1, colM1 in enumerate(self.geneMatrix1.T):
            for idx2, colM2 in enumerate(self.geneMatrix2.T):
                chiSquared = self.__computeCols__(colM1, colM2)
                self.correlations.append(sc.StatisticsCorrelation(idx1, idx2, chiSquared))

        valsFiltered = list(map(lambda correlation: correlation.correlationValue, self.correlations))
        Config.setChiSquaredThreshold(chiSquaredValues=valsFiltered)
        self.__classifySamples__()

    # returns chiSquared value
    def __computeCols__(self, colM1, colM2):
        expected = StatisticsUtilities.calculateExpected(colM1, colM2)
        observed = StatisticsUtilities.calculateObserved(colM1, colM2)

        return np.sum(np.divide(np.square(np.subtract(observed, expected)), expected))

    def __classifySamples__(self):
        for idx, sample in enumerate(self.correlations):
            self.correlations[idx].isInCorrelation(threshold=Config.CHI_SQUARED_THRESHOLD)

    def getResults(self):
        classes = np.array(list(map(lambda correlation: correlation.isCorrelated, self.correlations)))
        positive = (classes == True).sum()
        negative = (classes == False).sum()

        return {"values": self.correlations, "positive": positive, "negative": negative,
                "ratio": float(positive) / float(negative) * 100}

    def printResult(self):
        results = self.getResults()
        print("Positive: " + str(results["positive"]))
        print("Negative: " + str(results["negative"]))
        print("Ratio: " + str(float(results["positive"]) / float(results["negative"]) * 100) + " %")

    def validate(self):
        raise Exception("iStatistics - validate(self) not implemented")

    ## Tests

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
