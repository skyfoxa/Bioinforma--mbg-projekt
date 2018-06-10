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
        raise Exception("iStatistics - calculateObserved(self) not implemented")


    @staticmethod
    def getExpectedField(counts, mutation1, mutation2):
        mutInt1 = int(mutation1)
        mutInt2 = int(mutation2)
        return (counts[0][mutInt1] / (counts[0][0] + counts[0][1])) * \
                         (counts[1][mutInt2] / (counts[1][0] + counts[1][1])) * \
                         (counts[1][0] + counts[1][1])

    @staticmethod
    def calculateCountsInCols(col1, col2):
        def getCounts(col):
            unique, counts = np.unique(col, return_counts=True)
            return dict(zip(unique, counts))

        counts = np.empty((2,2), dtype=float)

        col1Counts = getCounts(col1)
        col2Counts = getCounts(col2)

        counts[0][0] = col1Counts[True]
        counts[0][1] = col1Counts[False]
        counts[1][0] = col2Counts[True]
        counts[1][1] = col2Counts[False]

        return counts


