#!/usr/bin/env python3

from .iFilter import iFilter
from .BooleanReferenceSeqFilter import BooleanReferenceSeqFilter
import numpy as np
from src.Configuration.applicationConfig import Config

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

# interface for filters used in Statistic tests
class LowMutationsFilter(iFilter):
    def __init__(self, data, threshold = Config.MUTATIONS_THRESHOLD):
        self.threshold = threshold
        super().__init__(data)

    def filterData(self):
        dataBool = (self.data>0).sum(0)>self.threshold
        return self.data[:, dataBool]