#!/usr/bin/env python3

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

class StatisticsCorrelation:

    firstSample = None
    secondSample = None
    correlationValue = None
    isCorrelated = None

    def __init__(self, firstSample, secondSample, correlationValue):
        self.firstSample = firstSample
        self.secondSample = secondSample
        self.correlationValue = correlationValue
        self.isCorrelated = False

    def isInCorrelation(self, threshold):
        self.isCorrelated = self.correlationValue > threshold


