import tensorflow as tf

def initializeModel(
    self,
    memorySize,
    windowSize,
    inputDimension,
    encoderStateSize,
    lstmStateSize,
    optimizer
):
    """Initialize the Model
    
    self: The object that called this method
    memorySize: Size of the memory (i.e. size of S and q),
    this is a scalar
    windowSize: Size of a window, this is a scalar
    threshold: Threshold value, above which an event is
    regarded as an extreme event
    inputDimension: Dimension of each input vector, it is a
    scalar
    hiddenStateSize: Dimension of the hidden state vector of
    the GRU, it is a scalar
    extremeValueIndex: Extreme Value Index parameter for the
    extreme value loss function, this is a scalar
    extremeLossWeight: Weight given to the extreme value loss,
    this is a scalar
    optimizer: The optimizer to be used for training the model

    Initialize the model parameters and hyperparameters of the
    model
    """
    self.memorySize = memorySize
    self.windowSize = windowSize
    self.optimizer = optimizer
    self.inputDimension = inputDimension
    self.encoderStateSize = encoderStateSize
    self.lstmStateSize = lstmStateSize
    self.memory = None
    self.q = None

    self.gruEncoder = tf.keras.layers.GRUCell(self.encoderStateSize)
    self.gruEncoder.build(input_shape = (self.inputDimension,))

    self.lstm = tf.keras.layers.LSTMCell(self.lstmStateSize)
    self.lstm.build(input_shape = (self.inputDimension,))

    self.b = tf.Variable(0)
