#!/usr/bin/env python3

from src.Statistics import *
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

class PermutationTest(iStatistics):
    plotTitle = None

    def __init__(self, geneMatrix1, geneMatrix2, testClass, plotTitle="PermutationTest"):
        super().__init__(geneMatrix1, geneMatrix2)
        self.test = globals()[testClass]

        if not issubclass(self.test, iStatistics):
            raise AttributeError("testClass must be subclass of iStatistics")

        self.plotTitle = plotTitle

    def __permutate__(self, geneMatrix):
        return np.random.permutation(geneMatrix)

    def compute(self):
        test = self.test(self.__permutate__(self.geneMatrix1), self.geneMatrix2, plotTitle=self.plotTitle)
        test.compute()


    def validate(self):
        raise Exception("iStatistics - validate(self) not implemented")

    def plot(self, values, title):
        plt.plot(values)
        plt.title(title)
        plt.savefig(title)
        plt.close()

        num_bins = 5
        n, bins, patches = plt.hist(values, num_bins, facecolor='blue', alpha=0.5)
        plt.savefig("Hist" + title)
        plt.close()
