import numpy as np
import os

from ts.data.univariate.nonexo import ArmaGenerator
from ts.model.univariate.deep import ExtremeTime
from ts.plot import Plot
from ts.log import GlobalLogger


def simpleData(n):
    P = Q = 5

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
    noiseGenParams = (100., 10.0)

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams) \
        .generate(n)


def main():
    GlobalLogger.getLogger().setLevel(2)

    n = 5500
    trainN = 5000
    horizon = 1

    targets = extremeData2(n)
    trainTargets = targets[:trainN]
    testTargets = targets[trainN:]

    Plot.plotDataCols(np.expand_dims(targets, axis=1))

    modelSavePath = os.path.expanduser('~/extreme.model')
    model = ExtremeTime(horizon, 80, 40, 10, 10, 0)

    losses = model.train(
        trainTargets,
        100,
        modelSavePath=modelSavePath,
        verboseLevel=2,
        returnLosses=True,
        numIterations=3
    )

    Plot.plotLoss(losses, xlabel='seq')

    loss, Ypred = model.evaluate(testTargets, returnPred=True)

    print(f'Test Loss Value: {loss}')
    Plot.plotPredTrue(Ypred, testTargets[horizon:])


if __name__ == '__main__':
    main()
