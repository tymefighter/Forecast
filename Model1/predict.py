def computeAttentionWeights(
    self
    state,
):
    return tf.squeeze(tf.nn.softmax(np.dot(
        self.S,
        state
    )))

def predictOneTimestep(
    self,
    gruState,
    X, 
    currTime
):
    nextState, _ = self.gru(np.expand_dims(X[i], 0), gruState)

    attentionWeights = self.computeAttentionWeights(nextState)
    extremePred = tf.math.reduce_sum(
        attentionWeights * self.q, 
        axis = 0
    )

    if Y[i] > self.epsilon:
        extremeTarget = 1
        numExtremeEvents += 1
    else:
        extremeTarget = 0
        numNormalEvents += 1

    yPred = semiPred + self.b * extremePred
    return yPred, nextState

def predictOutput(self, X):

    n = X.shape[0]
    state = self.gru.get_initial_state()
    Y = [None] * n

    for i in range(n):
        Y[i], state = self.predictOneTimestep(
            state,
            X,
            i
        )

    Y = np.array(Y)
    return Y