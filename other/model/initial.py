import tensorflow as tf

def initializeModel(
    self,
    memorySize,
    windowSize,
    threshold,
    inputDimension,
    hiddenStateSize,
    extremeValueIndex,
    extremeLossWeight,
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
    self.threshold = threshold
    self.optimizer = optimizer
    self.inputDimension = inputDimension
    self.hiddenStateSize = hiddenStateSize
    self.extremeValueIndex = extremeValueIndex
    self.extremeLossWeight = extremeLossWeight
    self.S = None
    self.q = None

    self.gru = tf.keras.layers.GRUCell(
        units = self.hiddenStateSize,
        activation = None,
        recurrent_activation = 'sigmoid'
    )
    self.gru.build(input_shape = (self.inputDimension,))

    self.out = tf.keras.layers.Dense(
        units = 1,
        activation = None
    )
    self.out.build(input_shape = (self.hiddenStateSize,))

    self.memOut = tf.keras.layers.Dense(
        units = 1,
        activation = 'sigmoid',
        input_shape = (self.hiddenStateSize,)
    )
    self.memOut.build(input_shape = (self.hiddenStateSize,))

    self.b = tf.Variable(0.0)
