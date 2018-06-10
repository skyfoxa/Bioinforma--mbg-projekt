import numpy as np
from .iFilter import iFilter


class BooleanReferenceSeqFilter(iFilter):
    def filterData(self, refSeq):
        return self.data == refSeq
