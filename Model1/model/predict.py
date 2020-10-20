import tensorflow as tf
import numpy as np

def computeAttentionWeights(
    self,
    state
):
    """Computes Attention Weights

    self: The object that called this method
    state: The state of the GRU with respect to which the attention
    parameters are to be calculated, shape: (H,)

    self.S is the constructed summary of memorySize sequences,
    hence it's dimension is (memorySize, H). Every row of this state
    is dot producted with state to get the similary between them,
    which gives us a vector of shape (memorySize, 1). Now, we take
    softmax over this vector to build the attention weights.

    Returns a vector of shape (memorySize,) which are the attention
    weights, i.e. if the returned vector is retVec, then retVec[i]
    is the attention weight corresponding to the ith sequence summary,
    self.S[i]
    """
    return tf.squeeze(tf.nn.softmax(tf.linalg.matmul(
        self.S,
        tf.expand_dims(state, axis = 1)
    )), axis = 1)

def predictOneTimestep(
    self,
    gruState,
    X, 
    currTime
):
    """Predict The Final Output for a Single Timestep

    self: The object that called this method
    gruState: Current state of the GRU, shape: (1, H)
    X: The entire input Sequence, it has shape (n, d)
    currTime: Current Timestep, it is a scalar

    Returns the following,
    - the final predicted output for current timestep,
    this is a scalar
    - The next state of the GRU, this has a shape (1, H)
    """
    nextState, _ = self.gru(np.expand_dims(X[i], 0), gruState)

    attentionWeights = \
        self.computeAttentionWeights(tf.squeeze(nextState))

    extremePred = tf.math.reduce_sum(
        attentionWeights * self.q, 
        axis = 0
    )

    yPred = semiPred + self.b * extremePred
    return yPred, nextState

def predictOutput(self, X):
    """Predict Output For an Input Sequence
    
    self: The object that called this method
    X: The entire input Sequence, it has shape (n, d)

    Returns a sequence of outputs for the corresponding input
    sequence, it has shape (n,)
    """

    n = X.shape[0]
    state = self.gru.get_initial_state(
        batch_size = 1, 
        dtype = tf.float32
    )

    Y = [None] * n
    for i in range(n):
        Y[i], state = self.predictOneTimestep(
            state,
            X,
            i
        )

    Y = np.array(Y)
    return Y