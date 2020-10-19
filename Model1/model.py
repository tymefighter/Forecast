from train import trainModel
from initial import initializeModel
from predict import predictOutput

class Model:

    def __init__(
        self,
        memorySize,
        windowSize,
        threshold,
        inputDimension,
        hiddenStateSize,
        extremeValueIndex,
        optimizer,
        extremeLossWeight,
        modelPath = None
    ):
        if modelPath is not None:
            self.loadModel(modelPath)
        else:
            self.initializeModel(
                memorySize,
                windowSize,
                threshold,
                inputDimension,
                hiddenStateSize,
                extremeValueIndex,
                extremeLossWeight,
                optimizer
            )

    def train(
        self,
        X, 
        Y, 
        seqLength,
        currTimestep = None
    ):
        self.trainModel(
            X, 
            Y, 
            seqLength,
            currTimestep
        )

    def predict(self, X):
        pass

    def saveModel(self, modelFilepath):
        pass

    def loadModel(self, modelFilepath):
        pass