import numpy as np
import tensorflow as tf
from ts.data.generate.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.special import ExtremeTime2
from ts.plot import Plot


def main():
    n = 21500
    trainN = 21000
    seqLength = 500
    numSeqPlot = 5

    trainData, testData = Utility.trainTestSplit(
        StandardGenerator('long_term').generate(n),
        trainN
    )

    # for i in range(numSeqPlot):
    #     Plot.plotDataCols(trainSequences[
    #         np.random.randint(0, len(trainSequences))
    #     ])

    model = ExtremeTime2(
        forecastHorizon=1,
        memorySize=20,
        windowSize=10,
        embeddingSize=10,
        contextSize=10
    )

    loss = model.train(
        targetSeries=trainData,
        sequenceLength=seqLength,
        numIterations=10,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                0.01,
                50,
                0.99
            )
        ),
        verboseLevel=1,
        returnLosses=True
    )

    Plot.plotLoss(loss)

    for i in range(numSeqPlot):
        idx = np.random.randint(0, trainN - seqLength)
        seq = trainData[idx:idx + seqLength]
        evalLoss, Ypred = model.evaluate(seq, returnPred=True)
        Ytrue = seq[1:]

        print(f'Train Eval Loss: {evalLoss}')
        Plot.plotPredTrue(Ypred, Ytrue, 'On Train')

    evalLoss, Ypred = model.evaluate(testData, returnPred=True)
    Ytrue = testData[1:]

    print(f'Test Eval Loss: {evalLoss}')
    Plot.plotPredTrue(Ypred, Ytrue, 'On Test')


if __name__ == '__main__':
    main()
