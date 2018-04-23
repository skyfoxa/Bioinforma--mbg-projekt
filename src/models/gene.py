#!/usr/bin/env python3

__authors__ = "Marek Zvara, Marek Hrvol"
__copyright__ = "Copyright 2018, Marek Zvara, Marek Hrvol"
__email__ = "zvaramar@fel.cvut.cz, hrvolmar@fel.cvut.cz"
__description__ = "MBG"

class Gene(object):
    def __init__(self, name, locationStart, locationEnd, chromosone):
        self.name = name
        self.locationStart = locationStart
        self.locationEnd = locationEnd
        self.chromosone = chromosone
        self.length = locationEnd - locationStart
