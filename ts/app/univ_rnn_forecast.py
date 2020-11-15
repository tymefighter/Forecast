import os
import numpy as np
import tensorflow as tf

from ts.data.univariate.nonexo import StandardGenerator
from ts.model.univariate.deep import SimpleRnnForecast
from ts.plot import Plot
from ts.utility import Utility


def main():
    data = StandardGenerator('simple').generate(5000)
    train, test = Utility.trainTestSplit(data, 4500)

    rnnForecast = SimpleRnnForecast(1, 10, 1, 0)
    trainSequences = Utility.breakSeq(train, 100)

    modelSavePath = os.path.expanduser('~/rnnModel')
    losses = rnnForecast.train(
        trainSequences,
        numIterations=2,
        optimizer=tf.optimizers.Adam(0.03),
        modelSavePath=modelSavePath
    )

    Plot.plotLoss(losses)

    loss, Ypred = rnnForecast.evaluate(test, None, True)
    Plot.plotPredTrue(Ypred, test[1:])


if __name__ == '__main__':
    main()
