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
        'optimizer' : self.optimizer,
        'inputDimension' : self.inputDimension,
        'encoderStateSize' : self.encoderStateSize,
        'lstmStateSize' : self.lstmStateSize,
        'memory' : self.memory,
        'q' : self.q,
        'gruEncoder' : self.gruEncoder.get_weights(),
        'lstm' : self.lstm.get_weights(),
        'W' : self.W.read_value(),
        'A' : self.memOut.A.read_value(),
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
    self.optimizer = saveDict['optimizer']
    self.inputDimension = saveDict['inputDimension']
    self.encoderStateSize = saveDict['encoderStateSize']
    self.lstmStateSize = saveDict['lstmStateSize']
    self.memory = saveDict['memory']
    self.q = saveDict['q']

    self.gruEncoder = tf.keras.layers.GRUCell(units = self.encoderStateSize)
    self.gruEncoder.build(input_shape = (self.inputDimension,))
    self.gruEncoder.set_weights(saveDict['gruEncoder'])

    self.lstm = tf.keras.layers.LSTMCell(units = self.encoderStateSize)
    self.lstm.build(input_shape = (self.inputDimension,))
    self.lstm.set_weights(saveDict['lstm'])

    self.W = tf.Variable(saveDict['W'])
    self.A = tf.Variable(saveDict['A'])
    self.b = tf.Variable(saveDict['b'])
