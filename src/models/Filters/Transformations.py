import numpy as np
from .iFilter import iFilter


class Transformations(object):

    @staticmethod
    def basesToBooleans(refSeq, data):
        return data != refSeq
