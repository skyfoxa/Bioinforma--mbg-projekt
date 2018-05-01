#!/usr/bin/env python3

import numpy as np
import urllib.request
import urllib.error

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

# Example to download particular data
# ./samtools view -b ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/alignment/HG00096.mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam 4:99792835-99851788 > HG00096.bam

class DataHandler(object):
    def __init__(self, gene1, gene2):
        self.gene1Path = gene1
        self.gene2Path = gene2

        self.__createArraysFrom__(gene1, gene2)
        self.checkIfSameGenes()
        self.__addPopulationsToData__()


    def __createArraysFrom__(self, gene1Path, gene2Path):
        gene1 = np.genfromtxt(gene1Path, delimiter='\t', dtype=str)
        gene2 = np.genfromtxt(gene2Path, delimiter='\t', dtype=str)
        self.gene1Info, self.gene1Data = self.__splitData__(gene1)
        self.gene2Info, self.gene2Data = self.__splitData__(gene2)
        self.gene1Data = self.__deleteLastColumns__(self.gene1Data)
        self.gene2Data = self.__deleteLastColumns__(self.gene2Data)


    def __splitData__(self, data, numberOfCols = 6):
        part2 = data[:,numberOfCols:]
        part1 = data[:,:numberOfCols]
        return part1, part2

    def __deleteLastColumns__(self, data, howMany = 1):
        return data[:,:-(1 + howMany)]

    def checkIfSameGenes(self):
        assert (self.gene1Info[:,1] == self.gene2Info[:,1]).all(), "Names of the individuals do not match!"

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
