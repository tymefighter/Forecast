import numpy as np
import tensorflow as tf

from ts.data.generate.univariate.nonexo import StandardGenerator
from ts.model.special import ExtremeTime2
from ts.plot import Plot
from ts.log import GlobalLogger


def main():
    GlobalLogger.getLogger().setLevel(1)

    n = 1500
    trainN = 1400
    horizon = 1

    targets = StandardGenerator('simple').generate(n)
    trainTargets = targets[:trainN]
    testTargets = targets[trainN:]

    Plot.plotDataCols(np.expand_dims(targets, axis=1))

    # modelSavePath = os.path.expanduser('~/extreme.model')
    modelSavePath = None
    model = ExtremeTime2(horizon, 10, 10, 20, 20, 0)

    losses = model.train(
        trainTargets,
        100,
        numIterations=1,
        optimizer=tf.optimizers.Adam(0.3),
        modelSavePath=modelSavePath,
        verboseLevel=2,
        returnLosses=True
    )

    Plot.plotLoss(losses, xlabel='seq')

    # model = ExtremeTime2(modelLoadPath=modelSavePath)
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
