class Config:
    MUTATIONS_THRESHOLD = 15
    CHI_SQUARED_THRESHOLD = 10.828
    SET_CHI_SQUARED_DYNAMICALLY = True
    CHI_SQUARED_EPSILON = 4
    VERBOSE = False
    SHOW_DELETED_VALUES = False
    LARGEST_POSSIBLE_SAMPLE = 150
    OUTPUT_FOLDER_PATH = "../output/"

    @staticmethod
    def setChiSquaredThreshold(chiSquaredValues):
        if not Config.SET_CHI_SQUARED_DYNAMICALLY:
            return
        Config.SET_CHI_SQUARED_DYNAMICALLY = False
        Config.CHI_SQUARED_THRESHOLD = 10.828 # TODO: logic for setting chi squared
        sortedVals = sorted(chiSquaredValues)
        minChiSquaredId = int(len(sortedVals) - len(sortedVals) / Config.LARGEST_POSSIBLE_SAMPLE)
        for i in range (minChiSquaredId, len(sortedVals)-5):
            if sortedVals[i+5] - sortedVals[i] > Config.CHI_SQUARED_EPSILON:
                print("CHI_SQUARED_THRESHOLD set to: " + str(sortedVals[i]))
                Config.CHI_SQUARED_THRESHOLD = sortedVals[i]
                return
        if sortedVals[minChiSquaredId] > Config.CHI_SQUARED_THRESHOLD:
            print("CHI_SQUARED_THRESHOLD set to DEFAULT: " + str(sortedVals[i]))
            Config.CHI_SQUARED_THRESHOLD = sortedVals[minChiSquaredId]




