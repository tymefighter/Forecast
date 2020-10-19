from train import trainModel
from initial import initializeModel
from predict import predictOutput
from saveLoad import saveModel, loadModel

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
        currTimestep = None,
        modelFilepath = None
    ):
        self.trainModel(
            X, 
            Y, 
            seqLength,
            currTimestep,
            modelFilepath
        )

    def predict(self, X):
        return self.predictOutput(X)

    def save(self, modelFilepath):
        self.saveModel(modelFilepath)

    def load(self, modelFilepath):
        self.loadModel(modelFilepath)