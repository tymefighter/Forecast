import numpy as np

from ts.data.generate.univariate.nonexo import StandardGenerator
from ts.model.special import ExtremeTime
from ts.plot import Plot
from ts.log import GlobalLogger


def main():
    GlobalLogger.getLogger().setLevel(1)

    n = 1500
    trainN = 1400
    horizon = 1

    targets = StandardGenerator('extreme_short').generate(n)
    trainTargets = targets[:trainN]
    testTargets = targets[trainN:]

    Plot.plotDataCols(np.expand_dims(targets, axis=1))

    # modelSavePath = os.path.expanduser('~/extreme.model')
    modelSavePath = None
    model = ExtremeTime(horizon, 10, 10, 10, 10, 0)

    losses = model.train(
        trainTargets,
        100,
        numIterations=1,
        modelSavePath=modelSavePath,
        verboseLevel=2,
        returnLosses=True
    )

    Plot.plotLoss(losses, xlabel='seq')

    # model = ExtremeTime(modelLoadPath=modelSavePath)
    loss, Ypred = model.evaluate(testTargets, returnPred=True)

    print(f'Test Loss Value: {loss}')
    Plot.plotPredTrue(Ypred, testTargets[horizon:])


if __name__ == '__main__':
    main()
