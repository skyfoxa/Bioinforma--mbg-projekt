class Config:
    MUTATIONS_THRESHOLD = 5
    CHI_SQUARED_THRESHOLD = 10.828
    SET_CHI_SQUARED_DYNAMICALLY = True
    CHI_SQUARED_EPSILON = 4


    def setChiSquaredThreshold(self, chiSquaredValues):
        if not self.SET_CHI_SQUARED_DYNAMICALLY:
            return
        self.SET_CHI_SQUARED_DYNAMICALLY = False
        self.CHI_SQUARED_THRESHOLD = 10.828 # TODO: logic for setting chi squared
        chiSquaredValues = chiSquaredValues.sort()
        for i in range (len(chiSquaredValues)-5):
            if chiSquaredValues[i+5] - chiSquaredValues[i] > self.CHI_SQUARED_EPSILON:
                print(chiSquaredValues[i])
                return




