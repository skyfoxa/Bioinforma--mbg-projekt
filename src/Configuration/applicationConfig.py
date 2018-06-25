class Config:
    MUTATIONS_THRESHOLD = 5
    CHI_SQUARED_THRESHOLD = 10.828
    SET_CHI_SQUARED_DYNAMICALLY = True
    CHI_SQUARED_EPSILON = 4

    @staticmethod
    def setChiSquaredThreshold(chiSquaredValues):
        if not Config.SET_CHI_SQUARED_DYNAMICALLY:
            return
        Config.SET_CHI_SQUARED_DYNAMICALLY = False
        Config.CHI_SQUARED_THRESHOLD = 10.828 # TODO: logic for setting chi squared
        sortedVals = sorted(chiSquaredValues)
        for i in range (len(sortedVals)-5):
            if sortedVals[i+5] - sortedVals[i] > Config.CHI_SQUARED_EPSILON:
                print(sortedVals[i])
                Config.CHI_SQUARED_THRESHOLD = sortedVals[i]
                return




