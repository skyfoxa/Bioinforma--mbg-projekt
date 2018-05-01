#!/usr/bin/env python3

import numpy as np

__authors__ = "Marek Zvara, Marek Hrvol"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz"
__description__ = "MBG"

# Example to download particular data
# ./samtools view -b ftp://ftp.1000genomes.ebi.ac.uk/vol1/ftp/phase3/data/HG00096/alignment/HG00096.mapped.ILLUMINA.bwa.GBR.low_coverage.20120522.bam 4:99792835-99851788 > HG00096.bam

class DataHandler(object):
    def __init__(self, gene1, gene2):
        self.gene1Path = gene1
        self.gene2Path = gene2

        self.__createArraysFrom__(gene1, gene2)


    def __createArraysFrom__(self, gene1Path, gene2Path):
        gene1Data = np.genfromtxt(gene1Path, delimiter='\t', dtype=str)
        gene2Data = np.genfromtxt(gene2Path, delimiter='\t', dtype=str)

