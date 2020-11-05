import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, TimeDistributed
import model
import cProfile

def generateArma(
    n,
    obsCoef,
    noiseCoef,
    noiseGenFunc,
    noiseGenParams,
    obsFunc = None,
    noiseFunc = None
):

    p = len(obsCoef)
    q = len(noiseCoef)    
    
    x = np.zeros(n)
    eps = np.zeros(n)

    for t in range(n):

        obsVal = 0
        for i in range(min(t, p)):
            obsVal += obsCoef[i] * x[t - i - 1]
        
        if obsFunc is not None:
            obsVal = obsFunc(obsVal)
        x[t] += obsVal
        
        noiseVal = 0
        for j in range(min(t, q)):
            noiseVal += noiseCoef[j] * eps[t - j - 1]

        if noiseFunc is not None:
            noiseVal = noiseFunc(noiseVal)
        x[t] += noiseVal

        eps[t] = noiseGenFunc(*noiseGenParams)
        x[t] += eps[t]

    return x

def genData():

    n = 100
    P = 5
    Q = 5

    obsCoef = np.concatenate([
        np.random.uniform(-0.1, 0, size = P // 2),
        np.random.uniform(0, 0.1, size = P // 2)
    ])

    noiseCoef = np.concatenate([
        np.random.uniform(-0.01, 0, size = Q // 2),
        np.random.uniform(0, 0.01, size = Q // 2)
    ])

    noiseGenFunc = np.random.gumbel
    noiseGenParams = (100., 10.0)

    trainSeq = generateArma(n, obsCoef, noiseCoef, noiseGenFunc, noiseGenParams)

    # print('Plotting Sequence')
    # plt.plot(trainSeq)
    # plt.show()

    x = trainSeq[:n-1, np.newaxis]
    y = trainSeq[1:, np.newaxis]

    return x, y

def main():

    x, y = genData()

    timeModel = model.Model(
        memorySize = 60,
        windowSize = 30,
        threshold = 120,
        inputDimension = 1,
        hiddenStateSize = 20,
        extremeValueIndex = 3.0,
        optimizer = tf.keras.optimizers.Adam(),
        extremeLossWeight = 2.0
    )

    cProfile.runctx(
        'timeModel.train(x, y, 100, verbose = 2)',
        {'timeModel' : timeModel, 'x' : x, 'y' : y},
        {}
    )

if __name__ == '__main__':
    main()