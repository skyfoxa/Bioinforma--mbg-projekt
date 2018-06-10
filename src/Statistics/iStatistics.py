#!/usr/bin/env python3

__authors__ = "Marek Zvara, Marek Hrvol, Filip Šamánek"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol, Filip Šamánek"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz, samanfil@fel.cvut.cz"
__description__ = "MBG"

class iStatistics(object):
    data = None

    def __init__(self, data):
        self.data = data

    def compute(self):
        raise Exception("iStatistics - compute(self) not implemented")

    def validate(self):
        raise Exception("iStatistics - validate(self) not implemented")

    def plot(self):
        raise Exception("iStatistics - plot(self) not implemented")

