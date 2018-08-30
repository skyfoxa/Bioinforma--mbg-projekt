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

    @staticmethod
    def getCorrelatedValues(samples):
        return list(filter(lambda sample: sample.isCorrelated, samples))

    @staticmethod
    def writeSamplesToFile(samples, fileName):
        with open(fileName + ".txt", "w") as f:
            firstSampleFirst = sorted(samples, key=lambda x: x.firstSample)
            secondSampleFirst = sorted(samples, key=lambda x: x.secondSample)

            f.write("Korelace prvniho genu s genem druhým. \n"
                    "Jednotlivé čísla poskytuje informace o pořadí sloupce v genu\n"
                    "za svistlítkem (|) je korelace druhého genu s prvním (pouze jiné pořadi, prvni sloupec je druhý gen)\n")
            for idOfSample, sample in enumerate(samples):
                f.write("{:<5}".format(str(firstSampleFirst[idOfSample].firstSample)) + " <----->   "
                        + "{:<5}".format(str(firstSampleFirst[idOfSample].secondSample)) + "     |     ")
                f.write("{:<5}".format(str(secondSampleFirst[idOfSample].secondSample)) + " <----->   "
                        + "{:<5}".format(str(secondSampleFirst[idOfSample].firstSample)) + "\n")



