import numpy as np
import os

from ts.data.univariate.nonexo import ArmaGenerator
from ts.model.univariate.deep import ExtremeTime
from ts.plot import Plot


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


def extremeData(n):

    P = Q = 5

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


def main():

    typeData = 'extreme'
    n = 50000

    if typeData == 'extreme':
        targets = extremeData(n)
    elif typeData == 'longTerm':
        targets = longTermData(n)
    else:
        targets = simpleData(n)

    Plot.plotDataCols(np.expand_dims(targets, axis=1))

    modelSavePath = os.path.expanduser('~/extreme.model')
    model = ExtremeTime(1, 80, 20, 10, 10, 0, logLevel=2)

    losses = model.train(
        targets,
        100,
        modelSavePath=modelSavePath,
        verboseLevel=2,
        logLevel=2,
        returnLosses=True
    )

    Plot.plotLoss(losses)


if __name__ == '__main__':
    main()
