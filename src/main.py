#!/usr/bin/env python3

import argparse
import os
import sys
sys.path.append('../')
from src.models import DataHandler
from src.Statistics import *
from src.models.Filters import *
from src.Statistics.statisticsUtilities import *
from src.models.statisticsCorrelation import StatisticsCorrelation as sc

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

def parseArgv():
    """Parse input parameters with argparse.

        Args:

        Returns:cd ..
            The return value. Returns object with input params:

    """
    parser = argparse.ArgumentParser(description='Gen correlation detector')

    parser.add_argument('-gene1', type=str,
                        help='Path to gene 1 - .ped file.')

    parser.add_argument('-gene2', type=str,
                        help='Path to gene 2 - .ped file.')

    args = parser.parse_args()

    if args.gene1 == None or args.gene2 == None:
        raise AttributeError("Specify correct path for genes data")

    if not os.path.exists(args.gene1):
        raise AttributeError("Specified path doesn't lead to .ped file (gene1) - " + str(args.gene1))

    if not os.path.exists(args.gene2):
        raise AttributeError("Specified path doesn't lead to .ped file (gene2) - " + str(args.gene2))

    return args

def tests():
    ChiSquared(None, None).testComputeCols()


def main():
    argv = parseArgv()
    dataHandler = DataHandler(gene1=argv.gene1, gene2=argv.gene2)
    gene1Filtered = Transformations.basesToBooleans(dataHandler.gene1Data, dataHandler.referenceGene1)
    gene2Filtered = Transformations.basesToBooleans(dataHandler.gene2Data, dataHandler.referenceGene2)

    gene1Filtered, indexesOfGene1 = LowMutationsFilter(gene1Filtered, dataHandler.indexesOfGene1).filterData()
    gene2Filtered, indexesOfGene2 = LowMutationsFilter(gene2Filtered, dataHandler.indexesOfGene2).filterData()

    print("###Chi Squared TEST")
    chiSquared = ChiSquared(gene1Filtered, gene2Filtered)
    chiSquared.compute()


    chiSquared.printResult()

    print("###Permutation TEST")
    permutationTest = PermutationTest(gene1Filtered, gene2Filtered, testClass=ChiSquared.__name__)
    permutationTest.compute()
    permutationTest.printResult()

    mappedSamples = dataHandler.mapSamplesToRealGeneCols(chiSquared.correlations)
    filteredMappedSamples = sc.getCorrelatedValues(mappedSamples)
    sc.writeSamplesToFile(filteredMappedSamples, fileName = dataHandler.gene1Name+" - "+dataHandler.gene2Name)

    StatisticsUtilities.outputResults(chiSquared.getResults(),
                                      permutationTest.getResults(),
                                      ChiSquared.__name__,
                                      PermutationTest.__name__,
                                      plotName=dataHandler.gene1Name+" - "+dataHandler.gene2Name
                                      )

if __name__ == "__main__":
    # tests()
    main()