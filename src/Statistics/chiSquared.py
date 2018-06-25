#!/usr/bin/env python3

from .iStatistics import iStatistics
from src.Statistics.statisticsUtilities import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.Configuration.applicationConfig import Config

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

class ChiSquared(iStatistics):
    plotTitle = None
    c1 = 0
    c2 = 0

    def __init__(self, geneMatrix1, geneMatrix2, plotTitle = "ChiSquared"):
        super().__init__(geneMatrix1, geneMatrix2)
        self.plotTitle = plotTitle

    def compute(self):
        vals = []
        for colM1 in self.geneMatrix1.T:
            for colM2 in self.geneMatrix2.T:
                chiSquared = self.__computeCols__(colM1, colM2)
                vals.append(chiSquared)

        Config.setChiSquaredThreshold(chiSquaredValues=vals)
        self.classifySamples(vals)

        self.printResult()
        self.plot(vals, self.plotTitle)



    # returns chiSquared value
    def __computeCols__(self, colM1, colM2):
        expected = StatisticsUtilities.calculateExpected(colM1, colM2)
        observed = StatisticsUtilities.calculateObserved(colM1, colM2)

        return np.sum(np.divide(np.square(np.subtract(observed, expected)), expected))

    def classifySamples(self, samples):
        for sample in samples:
            if sample > Config.CHI_SQUARED_THRESHOLD:
                self.c1 += 1
            else:
                self.c2 += 1

    def printResult(self):
        print("Positive: " + str(self.c1))
        print("Negative: " + str(self.c2))
        print("Ratio: " + str(float(self.c1) / float(self.c2) * 100) + " %")

    def validate(self):
        raise Exception("iStatistics - validate(self) not implemented")

    def plot(self, values, title):
        plt.plot(values)
        plt.title(title)
        plt.show()
        # plt.savefig(title)
        plt.close()

        #norm.rvs(10.0, 2.5, size=500)
        # Fit a normal distribution to the data:
        mu, std = norm.fit(values)

        # Plot the histogram.
        plt.hist(values, bins=50, alpha=0.6, color='g')

        # Plot the PDF.
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        p = norm.pdf(x, mu, std)
        plt.plot(x, p, 'k', linewidth=2)
        # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
        plt.title("Hist " + title)
        # plt.savefig("Hist" + title)
        plt.show()
        plt.close()

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