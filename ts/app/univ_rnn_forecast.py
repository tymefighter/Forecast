import os
import numpy as np
import tensorflow as tf

from ts.data.univariate.nonexo import ArmaGenerator
from ts.model.univariate.deep import SimpleRnnForecast
from ts.plot import Plot
from ts.utility import Utility


def simpleData(n):
    obsCoef = np.array([0.5, -0.2, 0.2])
    noiseCoef = np.array([0.5, -0.2])

    noiseGenFunc = np.random.normal
    noiseGenParams = (0.0, 1.0)

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams) \
        .generate(n)


def prepareData(data, trainSize):
    train = data[:trainSize]
    test = data[trainSize:]

    n = train.shape[0] - 1
    x = np.expand_dims(train[:n], axis=[0, 2])
    y = np.expand_dims(train[1:], axis=[0, 2])

    m = test.shape[0] - 1
    xt = np.expand_dims(test[:m], axis=[0, 2])
    yt = np.expand_dims(test[1:], axis=[0, 2])

    return x, y, xt, yt


def main():
    data = simpleData(5000)
    train, test = Utility.trainTestSplit(data, 4500)

    rnnForecast = SimpleRnnForecast(1, 10, 1, 0)
    trainSequences = Utility.breakSeq(train, 100)

    modelSavePath = os.path.expanduser('~/rnnModel')
    losses = rnnForecast.train(
        trainSequences,
        tf.optimizers.Adam(0.03),
        numIterations=2,
        modelSavePath=modelSavePath
    )

    Plot.plotLoss(losses)

    loss, Ypred = rnnForecast.evaluate(test, None, True)
    Plot.plotPredTrue(Ypred, test[1:])


if __name__ == '__main__':
    main()
