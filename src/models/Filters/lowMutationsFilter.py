#!/usr/bin/env python3

from .iFilter import iFilter
from .Transformations import Transformations
import numpy as np
from src.Configuration.applicationConfig import Config

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

# interface for filters used in Statistic tests
class LowMutationsFilter(iFilter):
    def __init__(self, data, indexesOfGene, threshold = Config.MUTATIONS_THRESHOLD):
        self.threshold = threshold
        super().__init__(data, indexesOfGene)

    def filterData(self):
        dataBoolMap = (self.data>0).sum(axis=0)>self.threshold
        return self.data[:, dataBoolMap], self.indexesOfGene[dataBoolMap]