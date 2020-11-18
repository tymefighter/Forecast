import numpy as np
import tensorflow as tf
from ts.data.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.univariate.oneseq.deep import ExtremeTime2
from ts.plot import Plot


def main():
    n = 20200
    trainN = 20000
    trainSeqLength = 200

    trainData, testData = Utility.trainTestSplit(
        StandardGenerator('long_term').generate(n),
        train=trainN
    )

    Plot.plotDataCols(trainData)

    model = ExtremeTime2(
        forecastHorizon=1,
        memorySize=10,
        windowSize=20,
        embeddingSize=20,
        contextSize=10
    )

    losses = model.train(
        targetSeries=trainData,
        sequenceLength=trainSeqLength,
        numIterations=30,
        optimizer=tf.keras.optimizers.Adam(0.03),
        verboseLevel=2,
        returnLosses=True
    )

    Plot.plotLoss(losses)

    evalLoss, Ypred = model.evaluate(trainData, returnPred=True)
    Ytrue = trainData[1:]
    print(f'Training Eval Loss: {evalLoss}')
    Plot.plotPredTrue(Ypred, Ytrue)

    evalLoss, Ypred = model.evaluate(testData, returnPred=True)
    Ytrue = testData[1:]
    print(f'Test Eval Loss: {evalLoss}')
    Plot.plotPredTrue(Ypred, Ytrue)


if __name__ == '__main__':
    main()
