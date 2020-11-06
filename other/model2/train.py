def runGruOnWindow(
    self,
    X,
    windowStartTime,
):
    """
    X: (n, d)
    return gruState: (encoderStateSize,)
    """
    gruState = self.getGruEncoderState()

    for t in range(
        windowStartTime, 
        windowStartTime + self.windowSize
    ):
        gruState, _ = self.gru(
            np.expand_dims(X[t], 0), 
            gruState
        )

    return tf.squeeze(gruState)

def buildMemory(
    self,
    X, 
    Y, 
    currentTime
):
    """
    X: (n, d)
    Y: (n,)
    """
    if currentTime < self.windowSize:
        raise Exception('Cannot Construct Memory')

    sampleLow = 0
    sampleHigh = currentTime - self.windowSize

    self.memory = [None] * self.memorySize
    self.q = [None] * self.memorySize

    for i in range(self.memorySize):
        windowStartTime = np.random.randint(
            sampleLow,
            sampleHigh + 1
        )

        self.memory[i] = self.runGruOnWindow(X, windowStartTime)
        self.q[i] = Y[windowStartTime + self.windowSize - 1]

    self.memory = tf.stack(self.memory)
    self.q = tf.convert_to_tensor(self.q, dtype = tf.float32)

def trainSequence(
    self,
    X, 
    Y, 
    seqStartTime, 
    seqEndTime
):
    """
    X: (n, d)
    Y: (n,)
    return: loss (1,)
    """
    with tf.GradientTape() as tape:
        self.buildMemory(self, X, seqStartTime)

        lstmStateList = self.getLstmStates()

        yPredSeq = []
        for t in range(seqStartTime, seqEndTime + 1):
            yPred, stateList = self.predictTimestep(lstmStateList, X, t)
            yPredSeq.append(yPred)

        yPredSeq = \
            tf.convert_to_tensor(yPredSeq, dtype = tf.float32)

        loss = tf.keras.losses.MSE(
            yPredSeq, 
            Y[seqStartTime : seqEndTime + 1]
        )
    
    trainableVars = self.gruEncoder.trainable_variables \
        + self.lstm.trainable_variables \
        + [self.W, self.A, self.b]

    grads = tape.gradient(loss, trainableVars)
    self.optimizer.apply_gradients(zip(
        grads,
        trainableVars
    ))

    return loss

def trainModel(
    self, 
    X, 
    Y,
    seqLength,
    currTimestep,
    modelFilepath,
    verbose
):
    seqStartTime = self.windowSize
    if currTimestep is not None:
        seqStartTime = max(seqStartTime, currTimestep)

    n = X.shape[0]
    if seqStartTime >= n:
        raise Exception("Insufficient Data")

    while seqStartTime < n:
        seqEndTime = min(
            n - 1, 
            seqStartTime + seqLength - 1
        )
        
        startTime = time.time()

        loss = self.trainSequence(X, Y, seqStartTime, seqEndTime)

        endTime = time.time()
        timeTaken = endTime - startTime
        if verbose > 0:
            print(
                f'start timestep: {seqStartTime}' \
                + f' | end timestep: {seqEndTime} ' \
                + f' | time taken: {timeTaken : .2f} sec' \
                + f' | Loss: {loss}'
            )

        seqStartTime += seqLength

        if modelFilepath is not None:
            self.saveModel(modelFilepath)

    self.buildMemory(X, Y, n - 1)
