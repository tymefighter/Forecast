import pickle

def saveModel(self, modelFilepath):
    """Save the Model Information

    self: The object that called this method
    modelFilepath: Path where to save the model information
    """

    if self.S is None:
        raise Exception('Memory not constructed - cannot save model')
    
    saveDict = {
        'memorySize' : self.memorySize,
        'windowSize' : self.windowSize,
        'threshold' : self.threshold,
        'optimizer' : self.optimizer,
        'inputDimension' : self.inputDimension,
        'hiddenStateSize' : self.hiddenStateSize,
        'extremeValueIndex' : self.extremeValueIndex,
        'extremeLossWeight' : self.extremeLossWeight,
        'S' : self.S,
        'q' : self.q,
        'gru' : self.gru.get_weights(),
        'out' : self.out.get_weights(),
        'memOut' : self.memOut.get_weights(),
        'b' : self.b.read_value()
    }

    fl = open(modelFilepath, 'wb')
    pickle.dump(saveDict, fl)
    fl.close()

def loadModel(self, modelFilepath):
    """Load the Model Information

    self: The object that called this method
    modelFilepath: Path from where to load the model
    information
    """
    
    fl = open(modelFilepath, 'rb')
    saveDict = pickle.load(fl)
    fl.close()

    self.memorySize = saveDict['memorySize']
    self.windowSize = saveDict['windowSize']
    self.threshold = saveDict['threshold']
    self.optimizer = saveDict['optimizer']
    self.inputDimension = saveDict['inputDimension']
    self.hiddenStateSize = saveDict['hiddenStateSize']
    self.extremeValueIndex = saveDict['extremeValueIndex']
    self.extremeLossWeight = saveDict['extremeLossWeight']
    self.S = saveDict['S']
    self.q = saveDict['q']
    self.gru.set_weights(saveDict['gru'])
    self.out.set_weights(saveDict['out'])
    self.memOut.set_weights(saveDict['memOut'])
    self.b.assign(saveDict['b'])