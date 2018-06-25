#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from src.Configuration.applicationConfig import Config
__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

class StatisticsUtilities(object):

    @staticmethod
    def calculateExpected(col1, col2): #TODO: rename variables names
        counts = StatisticsUtilities.calculateCountsInCols(col1, col2)
        expected = np.empty((2,2), dtype=float)

        for i in range(2):
            for j in range(2):
                expected[i][j] = StatisticsUtilities.getExpectedField(counts,i,j)

        return expected


    @staticmethod
    def calculateObserved(col1, col2):
        observed = np.empty((2, 2), dtype=float)

        observed[0][0] = np.logical_and(col1, col2).sum()
        observed[0][1] = np.greater(col1, col2).sum()
        observed[1][0] = np.less(col1, col2).sum()
        observed[1][1] = np.logical_not(np.logical_or(col1, col2)).sum()

        return observed


    @staticmethod
    def getExpectedField(counts, mutation1, mutation2):
        return (counts[0][mutation1] * counts[1][mutation2]) / (counts[0][0] + counts[0][1])


    @staticmethod
    def calculateCountsInCols(col1, col2):
        counts = np.empty((2,2), dtype=float)

        col1Counts = StatisticsUtilities.__getCounts__(col1)
        col2Counts = StatisticsUtilities.__getCounts__(col2)

        counts[0][0] = col1Counts[True]
        counts[0][1] = col1Counts[False]
        counts[1][0] = col2Counts[True]
        counts[1][1] = col2Counts[False]

        return counts

    @staticmethod
    def __getCounts__(col):
        d = {}
        d[True] = col.sum()
        d[False] = len(col) - d[True]

        return d

    @staticmethod
    def plotChiSquaredValues():

        StatisticsUtilities.__saveOrShowPlot__("omg")

    @staticmethod
    def __saveOrShowPlot__(name):
        if Config.VERBOSE:
            plt.show()
        else:
            plt.savefig(name)
        plt.close()
