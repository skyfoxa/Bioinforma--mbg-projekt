#!/usr/bin/env python3

import numpy as np

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

class StatisticsUtilities(object):

    @staticmethod
    def calculateExpected(col1, col2): #TODO: rename variables names
        counts = StatisticsUtilities.calculateCountsInCols(col1, col2)
        expected = np.empty((2,2), dtype=float)
        expected[0][0] = StatisticsUtilities.getExpectedField(counts,False,False)

        expected[0][1] = StatisticsUtilities.getExpectedField(counts,False,True)

        expected[1][0] = StatisticsUtilities.getExpectedField(counts,True,False)

        expected[1][1] = StatisticsUtilities.getExpectedField(counts,True,True)

        return expected


    @staticmethod
    def calculateObserved(col1, col2):
        observed = np.empty((2, 2), dtype=float)

        observed[0][0] = StatisticsUtilities.__getCounts__(np.logical_and(col1, col2))[True]
        observed[0][1] = StatisticsUtilities.__getCounts__(np.greater(col1, col2))[True]
        observed[1][0] = StatisticsUtilities.__getCounts__(np.less(col1, col2))[True]
        observed[1][1] = StatisticsUtilities.__getCounts__(np.logical_and(np.logical_not(col1), np.logical_not(col2)))[True]

        return observed


    @staticmethod
    def getExpectedField(counts, mutation1, mutation2):
        mutInt1 = int(mutation1)
        mutInt2 = int(mutation2)
        return (counts[0][mutInt1] / (counts[0][0] + counts[0][1])) * \
                         (counts[1][mutInt2] / (counts[1][0] + counts[1][1])) * \
                         (counts[1][0] + counts[1][1])

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
        unique, counts = np.unique(col, return_counts=True)

        d = dict(zip(unique, counts))

        if True not in d:
            d[True] = 0

        if False not in d:
            d[False] = 0

        return d


