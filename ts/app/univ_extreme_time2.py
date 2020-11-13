import numpy as np
import tensorflow as tf
import os

from ts.data.univariate.nonexo import ArmaGenerator
from ts.model.univariate.deep import ExtremeTime2
from ts.plot import Plot


def simpleData1(n):
    obsCoef = np.array([0.5, -0.2, 0.2])
    noiseCoef = np.array([0.5, -0.2])

    noiseGenFunc = np.random.normal
    noiseGenParams = (0.0, 1.0)

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams) \
        .generate(n)


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

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams) \
        .generate(n)


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

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams) \
        .generate(n)


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

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams) \
        .generate(n)


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

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams) \
        .generate(n)


def main():
    n = 1500
    trainN = 1400
    horizon = 1

    targets = simpleData1(n)
    trainTargets = targets[:trainN]
    testTargets = targets[trainN:]

    Plot.plotDataCols(np.expand_dims(targets, axis=1))

    model = ExtremeTime2(horizon, 10, 10, 20, 20, 0)

    losses = model.train(
        trainTargets,
        100,
        optimizer=tf.optimizers.Adam(0.3),
        verboseLevel=2,
        returnLosses=True,
        numIterations=5
    )

    Plot.plotLoss(losses, xlabel='seq')

    loss, Ypred = model.evaluate(trainTargets, returnPred=True)
    Ytrue = trainTargets[horizon:]

    print(f'Train Loss Value: {loss}')
    Plot.plotPredTrue(Ypred, Ytrue)

    loss, Ypred = model.evaluate(testTargets, returnPred=True)
    Ytrue = testTargets[horizon:]

    print(f'Test Loss Value: {loss}')
    Plot.plotPredTrue(Ypred, Ytrue)


if __name__ == '__main__':
    main()