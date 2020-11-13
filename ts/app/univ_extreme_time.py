import numpy as np
import tensorflow as tf
import os

from ts.data.univariate.nonexo import ArmaGenerator
from ts.model.univariate.deep import ExtremeTime
from ts.plot import Plot


def simpleData1(n):
    obsCoef = np.array([0.5, -0.2, 0.2])
    noiseCoef = np.array([0.5, -0.2])

    noiseGenFunc = np.random.normal
    noiseGenParams = (0.0, 1.0)

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams, logLevel=2) \
        .generate(n, logLevel=2)


def simpleData(n):
    P = Q = 20

    obsCoef = np.concatenate([
        np.random.uniform(-0.1, 0, size=P // 2),
        np.random.uniform(0, 0.1, size=P // 2)
    ])

    noiseCoef = np.concatenate([
        np.random.uniform(-0.01, 0, size=Q // 2),
        np.random.uniform(0, 0.01, size=Q // 2)
    ])

    noiseGenFunc = np.random.normal
    noiseGenParams = (0.0, 1.0)

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams, logLevel=2) \
        .generate(n, logLevel=2)


def longTermData(n):
    P = Q = 50

    obsCoef = np.concatenate([
        np.random.uniform(-0.1, 0, size=P // 2),
        np.random.uniform(0, 0.1, size=P // 2)
    ])

    noiseCoef = np.concatenate([
        np.random.uniform(-0.01, 0, size=Q // 2),
        np.random.uniform(0, 0.01, size=Q // 2)
    ])

    noiseGenFunc = np.random.normal
    noiseGenParams = (10.0, 1.0)

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams, logLevel=2) \
        .generate(n, logLevel=2)


def extremeData1(n):
    P = Q = 10

    obsCoef = np.concatenate([
        np.random.uniform(-0.1, 0, size=P // 2),
        np.random.uniform(0, 0.1, size=P // 2)
    ])

    noiseCoef = np.concatenate([
        np.random.uniform(-0.01, 0, size=Q // 2),
        np.random.uniform(0, 0.01, size=Q // 2)
    ])

    noiseGenFunc = np.random.lognormal
    noiseGenParams = (1.0, 1.0)

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams, logLevel=2) \
        .generate(n, logLevel=2)


def extremeData2(n):
    P = Q = 10

    obsCoef = np.concatenate([
        np.random.uniform(-0.1, 0, size=P // 2),
        np.random.uniform(0, 0.1, size=P // 2)
    ])

    noiseCoef = np.concatenate([
        np.random.uniform(-0.01, 0, size=Q // 2),
        np.random.uniform(0, 0.01, size=Q // 2)
    ])

    noiseGenFunc = np.random.gumbel
    noiseGenParams = (-5., 10.0)

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams, logLevel=2) \
        .generate(n, logLevel=2)


def main():
    n = 1500
    trainN = 1400
    horizon = 1

    targets = extremeData2(n)
    trainTargets = targets[:trainN]
    testTargets = targets[trainN:]

    Plot.plotDataCols(np.expand_dims(targets, axis=1))

    model = ExtremeTime(horizon, 20, 20, 20, 20, 0, logLevel=2)

    losses = model.train(
        trainTargets,
        100,
        optimizer=tf.optimizers.Adam(0.7),
        verboseLevel=2,
        logLevel=2,
        returnLosses=True,
        numIterations=5
    )

    Plot.plotLoss(losses, xlabel='seq')

    loss, Ypred = model.evaluate(trainTargets, logLevel=2, returnPred=True)
    Ytrue = trainTargets[horizon:]

    print(f'Train Loss Value: {loss}')
    Plot.plotPredTrue(Ypred, Ytrue)

    loss, Ypred = model.evaluate(testTargets, logLevel=2, returnPred=True)
    Ytrue = testTargets[horizon:]

    print(f'Test Loss Value: {loss}')
    Plot.plotPredTrue(Ypred, Ytrue)


if __name__ == '__main__':
    main()
