#!/usr/bin/env python3

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

# interface for filters used in Statistic tests
class iFilter(object):
    data = None
    indexesOfGene = None

    def __init__(self, data, indexesOfGene):
        self.data = data
        self.indexesOfGene = indexesOfGene

    def filterData(self):
        raise Exception("iFilter - filterData(self) must be implemented")

