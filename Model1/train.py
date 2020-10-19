import tensorflow as tf
import numpy as np

def runGruOnWindow(
    self,
    X,
    windowStartTime,
):
    state = self.gru.get_initial_state()
    for t in range(
        windowStartTime, 
        windowStartTime + self.windowSize
    ):
        state, _ = self.gru(X[t])

    return state

def buildMemory(
    self,
    X, 
    Y, 
    currentTime
):
    if currentTime < self.windowSize:
        raise Exception("Cannot Construct Memory")

    sampleLow = 0
    sampleHigh = currentTime - self.windowSize

    self.S = [None] * self.memorySize
    self.q = [None] * self.memorySize

    for i in range(self.memorySize):
        windowStartTime = np.random.randint(
            sampleLow,
            sampleHigh + 1
        )

        self.S[i] = self.runGruOnWindow(X, windowStartTime)
        if Y[windowStartTime + self.memorySize] > self.epsilon:
            self.q[i] = 1
        else:
            self.q[i] = 0

    
    self.S = np.array(self.S)
    self.q = np.array(self.q)

def trainOneSeq(
    self, 
    X, 
    Y, 
    seqStartTime, 
    seqEndTime
):
    pass

def trainModel(
    self, 
    X, 
    Y,
    seqLength,
    modelFilepath = None,
    currSeq = None
):
    pass