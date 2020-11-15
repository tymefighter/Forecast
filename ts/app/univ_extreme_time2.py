import numpy as np
import tensorflow as tf

from ts.data.univariate.nonexo import StandardGenerator
from ts.model.univariate.oneseq.deep import ExtremeTime2
from ts.plot import Plot


def main():
    n = 1500
    trainN = 1400
    horizon = 1

    targets = StandardGenerator('simple').generate(n)
    trainTargets = targets[:trainN]
    testTargets = targets[trainN:]

    Plot.plotDataCols(np.expand_dims(targets, axis=1))

    modelSavePath = None  # os.path.expanduser('~/extreme2.model')
    model = ExtremeTime2(horizon, 10, 10, 20, 20, 0)

    losses = model.train(
        trainTargets,
        100,
        numIterations=5,
        optimizer=tf.optimizers.Adam(0.3),
        modelSavePath=modelSavePath,
        verboseLevel=2,
        returnLosses=True
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