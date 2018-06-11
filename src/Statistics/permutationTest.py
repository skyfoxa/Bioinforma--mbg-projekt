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

    def __init__(self, geneMatrix1, geneMatrix2):
        super().__init__(geneMatrix1, geneMatrix2)

    def __permutate__(self, geneMatrix):
        return np.random.permutation(geneMatrix)

    def compute(self):
        chiSquared = ChiSquared(self.__permutate__(self.geneMatrix1), self.geneMatrix2, plotTitle="PermutationTest")
        chiSquared.compute()


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
