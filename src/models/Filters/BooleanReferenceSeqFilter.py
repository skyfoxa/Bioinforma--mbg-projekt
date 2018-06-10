import numpy as np
from .iFilter import iFilter


class BooleanReferenceSeqFilter(iFilter):
    refSeq = None

    def __init__(self, data, refSeq):
        super().__init__(data)
        self.refSeq = refSeq

    def filterData(self):
        return self.data != self.refSeq
