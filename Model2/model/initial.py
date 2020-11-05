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

    self.W = tf.Variable(tf.random.normal((1, self.lstmStateSize)))
    self.A = tf.Variable(
        tf.random.normal((self.encoderStateSize, self.lstmStateSize))
    )

    self.b = tf.Variable(0)

def getLstmStates(self):
    return self.lstm.get_initial_state(
        batch_size = 1,
        dtype = tf.float32
    )

def getGruEncoderState(self):

    return self.gruEncoder.get_initial_state(
        batch_size = 1, 
        dtype = tf.float32
    )
