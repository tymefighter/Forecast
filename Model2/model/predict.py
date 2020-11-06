import tensorflow as tf
import numpy as np

def computeAttentionWeights(
    self,
    embedding
):
    """
    embedding: (encoderStateSize,)
    return: (memorySize,)
    """
    return tf.squeeze(tf.nn.softmax(tf.linalg.matmul(
        self.S,
        tf.expand_dims(embedding, axis = 1)
    ), axis = 0))

def predictTimestep(
    self,
    lstmStateList,
    X, 
    currTime
):
    """
    lstmStateList = [lstmHiddenState, lstmCellState]
    lstmHiddenState: (1, lstmStateSize)
    lstmCellState: (1, lstmStateSize)
    X: (n, d)
    return: (1,), next stateList (same dim as lstmStateList)
    """
    
    [lstmHiddenState, lstmCellState] = self.lstm(
        X[currTime],
        lstmStateList
    )

    embedding = tf.matmul(
        self.A, 
        tf.expand_dims(tf.squeeze(lstmHiddenState), axis = 1)
    )
    attentionWeights = self.computeAttentionWeights(embedding)

    o1 = tf.squeeze(tf.matmul(
        self.W, 
        tf.expand_dims(tf.squeeze(lstmHiddenState), axis = 1)
    ))

    o2 = tf.reduce_sum(attentionWeights * self.q)

    bSigmoid = tf.nn.sigmoid(self.b)
    return bSigmoid * o1 + (1 - bSigmoid) * o2, [lstmHiddenState, lstmCellState]

def predictOutput(self, X):
    """Predict Output For an Input Sequence
    
    self: The object that called this method
    X: The entire input Sequence, it has shape (n, d)

    Returns a sequence of outputs for the corresponding input
    sequence, it has shape (n,)
    """

    n = X.shape[0]
    stateList = self.getInitialStates()
    yPred = [None] * n

    for i in range(n):
        yPred[i], stateList = \
            self.predictTimestep(stateList, X, i)

    yPred = np.array(yPred)
    return yPred
