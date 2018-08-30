#!/usr/bin/env python3

import urllib.error
import urllib.request

import numpy as np

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"


# Example to download particular data
# ./samtools view -b ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/alignment/HG00096.mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam 4:99792835-99851788 > HG00096.bam

class DataHandler(object):
    populations = None
    gene1Data = None
    gene2Data = None
    gene1Info = None
    gene2Info = None
    referenceGene1 = None
    referenceGene2 = None
    gene1Name = None
    gene2Name = None
    indexesOfGene1 = None
    indexesOfGene2 = None

    def __init__(self, gene1, gene2):
        self.gene1Data, self.gene1Info, self.referenceGene1, self.gene1Name = self.__initGene__(gene1)
        self.gene2Data, self.gene2Info, self.referenceGene2, self.gene2Name = self.__initGene__(gene2)

        # Creates vector of numbers in range <1, numberOfCols>
        self.indexesOfGene1 = np.array(range(1, self.gene1Data.shape[1] + 1))
        self.indexesOfGene2 = np.array(range(1, self.gene2Data.shape[1] + 1))

        self.checkIfSameGenes()
        # self.__addPopulationsToData__()

    def __initGene__(self, genePath):
        geneInfo, geneData = self.__createArraysFrom__(genePath)
        referenceGene = self.__getReferenceSeq__(geneData)
        geneName = genePath.split("/")[-1]
        return geneData, geneInfo, referenceGene, geneName

    def __createArraysFrom__(self, genePath):
        gene = np.genfromtxt(genePath, delimiter='\t', dtype=str)
        geneInfo, self.geneData = self.__splitData__(gene)
        geneData = self.__deleteLastColumns__(self.geneData)
        return geneInfo, geneData

    def __splitData__(self, data, numberOfCols=6):
        part2 = data[:, numberOfCols:]
        part1 = data[:, :numberOfCols]
        return part1, part2

    def __deleteLastColumns__(self, data, howMany=1):
        return data[:, :-(1 + howMany)]

    def checkIfSameGenes(self):
        assert (self.gene1Info[:, 1] == self.gene2Info[:, 1]).all(), "Names of the individuals do not match!"

    def __getReferenceSeq__(self, data):
        return np.apply_along_axis(self.__getGreatestAmount__, 0, data)

    def __getGreatestAmount__(self, vector):
        (values, counts) = np.unique(vector, return_counts=True)
        ind = np.argmax(counts)
        return values[ind]

    def __addPopulationsToData__(self):
        self.fetchPopulations()

        if len(self.populations.keys()) != 0:
            self.gene1Info = np.apply_along_axis(self.__addPopulation__, 1, self.gene1Info)
            self.gene2Info = np.apply_along_axis(self.__addPopulation__, 1, self.gene2Info)

    def __addPopulation__(self, row):
        population = self.populations[row[1]]
        return np.append(row, [population])

    # Returns dict - Sample_Name : Population
    def fetchPopulations(self):
        urlPopulations = "ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/20130502.phase3.analysis.sequence.index"
        req = urllib.request.Request(urlPopulations)
        try:
            with urllib.request.urlopen(req) as response:

                isHeader = True

                self.populations = {}
                nameIndex = -1
                popIndex = -1

                for line in response.readlines():
                    data = line.decode('UTF-8').strip().split('\t')
                    if isHeader:
                        items = list(map(lambda item: item.lower(), data))
                        popIndex = items.index('population')
                        nameIndex = items.index('sample_name')
                        isHeader = False
                    else:
                        self.populations[data[nameIndex]] = data[popIndex]

                return self.populations
        except urllib.error.HTTPError as e:
            print(e.reason)
        except urllib.error.URLError as e:
            print(e.reason)
        except:
            print("Fetch populations: Unexpected error:")

        return None
