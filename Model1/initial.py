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
        recurrent_activation = 'sigmoid',
        input_shape = (self.inputDimension,)
    )

    self.out = tf.keras.layers.Dense(
        units = 1,
        activation = None,
        input_shape = (self.hiddenStateSize,)
    )

    self.memOut = tf.keras.layers.Dense(
        units = 1,
        activation = 'sigmoid',
        input_shape = (self.hiddenStateSize,)
    )

    self.b = tf.Variable(0)
