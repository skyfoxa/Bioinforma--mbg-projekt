#!/usr/bin/env python3

from .iFilter import iFilter

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

# interface for filters used in Statistic tests
class LowMutationsFilter(iFilter):
    def __init__(self, data):
        iFilter.__init__(self, data)

    def filterData(self):
        return self.data