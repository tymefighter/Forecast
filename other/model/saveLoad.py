import tensorflow as tf
import numpy as np
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

    self.gru = tf.keras.layers.GRUCell(
        units = self.hiddenStateSize,
        activation = None,
        recurrent_activation = 'sigmoid'
    )
    self.gru.build(input_shape = (self.inputDimension,))
    self.gru.set_weights(saveDict['gru'])

    self.out = tf.keras.layers.Dense(
        units = 1,
        activation = None
    )
    self.out.build(input_shape = (self.hiddenStateSize,))
    self.out.set_weights(saveDict['out'])

    self.memOut = tf.keras.layers.Dense(
        units = 1,
        activation = 'sigmoid',
        input_shape = (self.hiddenStateSize,)
    )
    self.memOut.build(input_shape = (self.hiddenStateSize,))
    self.memOut.set_weights(saveDict['memOut'])

    self.b = tf.Variable(saveDict['b'])