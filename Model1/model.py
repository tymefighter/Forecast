class Model:

    def __init__(
        self,
        memorySize,
        windowSize,
        threshold,
        optimizer,
        modelPath = None
    ):
        pass

    def train(
        self,
        X, 
        Y, 
        seqLength,
        modelFilepath = None,
        currSeq = None
    ):
        pass

    def predict(self, X):
        pass

    def saveModel(self, modelFilepath):
        pass

    def loadModel(self, modelFilepath):
        pass