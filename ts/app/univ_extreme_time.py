import numpy as np
import os

from ts.data.univariate.nonexo import StandardGenerator
from ts.model.univariate.deep import ExtremeTime
from ts.plot import Plot
from ts.log import GlobalLogger


def main():
    GlobalLogger.getLogger().setLevel(2)

    n = 1500
    trainN = 1400
    horizon = 1

    targets = StandardGenerator('extreme_short').generate(n)
    trainTargets = targets[:trainN]
    testTargets = targets[trainN:]

    Plot.plotDataCols(np.expand_dims(targets, axis=1))

    modelSavePath = None  # os.path.expanduser('~/extreme.model')
    model = ExtremeTime(horizon, 10, 10, 10, 10, 0)

    losses = model.train(
        trainTargets,
        100,
        numIterations=3,
        modelSavePath=modelSavePath,
        verboseLevel=2,
        returnLosses=True
    )

    Plot.plotLoss(losses, xlabel='seq')

    loss, Ypred = model.evaluate(testTargets, returnPred=True)

    print(f'Test Loss Value: {loss}')
    Plot.plotPredTrue(Ypred, testTargets[horizon:])


if __name__ == '__main__':
    main()
