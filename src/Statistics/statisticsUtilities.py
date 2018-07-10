#!/usr/bin/env python3

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from src.Configuration.applicationConfig import Config

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"


class StatisticsUtilities(object):
    @staticmethod
    def calculateExpected(col1, col2):  # TODO: rename variables names
        counts = StatisticsUtilities.calculateCountsInCols(col1, col2)
        expected = np.empty((2, 2), dtype=float)

        for i in range(2):
            for j in range(2):
                expected[i][j] = StatisticsUtilities.getExpectedField(counts, i, j)

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
        counts = np.empty((2, 2), dtype=float)

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
    def outputResults(result1, result2, result1Name, result2Name, plotName):
        result1Vals = result1["values"]
        result2Vals = result2["values"]
        if not Config.SHOW_DELETED_VALUES:
            result1Vals = list(filter(lambda val: val[2] >= Config.CHI_SQUARED_THRESHOLD, result1Vals))
            result2Vals = list(filter(lambda val: val[2] >= Config.CHI_SQUARED_THRESHOLD, result2Vals))
        else:
            plotName = plotName + "_withDeleted"

        positive1 = result1["positive"]
        negative1 = result1["negative"]
        ratio1 = result1["ratio"]

        positive2 = result2["positive"]
        negative2 = result2["negative"]
        ratio2 = result2["ratio"]

        plotName = plotName + ".png"

        fig, axs = plt.subplots(3, 1)

        cols = ('Positive', 'Negative', 'Ratio')
        cellText = np.array([[positive1, negative1, ratio1], [positive2, negative2, ratio2]]).reshape((2, 3))

        axs[0].axis('tight')
        axs[0].axis('off')
        the_table = axs[0].table(cellText=cellText,
                                 rowLabels=[result1Name, result2Name],
                                 colLabels=cols)

        plt.subplots_adjust(left=0.25, hspace=0.5)

        StatisticsUtilities.__plotHistogram__(axs[1], list(map(lambda val: val[2], result1Vals)), 'r', 'g+',
                                              result1Name)
        StatisticsUtilities.__plotHistogram__(axs[2], list(map(lambda val: val[2], result2Vals)), 'b', 'y-',
                                              result2Name)
        # StatisticsUtilities.__saveOrShowPlot__(plotName)

        # plt.show()
        plt.savefig(plotName)

    @staticmethod
    def __plotHistogram__(axs, values, histColor, fitLineColor, name):
        # Plot the PDF.
        if len(values) != 0:
            n, bins, patches = axs.hist(values, density=True, bins=50, alpha=0.6, color=histColor, edgecolor='black',
                                        linewidth=1, label="Histogram - " + name)

            mu, std = norm.fit(values)
            y = norm.pdf(bins, mu, std)
            axs.plot(bins, y, fitLineColor, linewidth=2,  label="Best fit line - " + name)
        else:
            axs.hist([Config.CHI_SQUARED_THRESHOLD], density=True, alpha=0, color='white', label="Histogram - " + name)


        StatisticsUtilities.__addLegend__(axs)

    @staticmethod
    def __addLegend__(axs):
        legend = axs.legend(loc='upper center')
        frame = legend.get_frame()
        frame.set_facecolor('0.90')

    @staticmethod
    def __saveOrShowPlot__(name):
        if Config.VERBOSE:
            plt.show()
        else:
            plt.savefig(name)
        plt.close()
