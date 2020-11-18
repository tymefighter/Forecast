import numpy as np
import tensorflow as tf
from ts.data.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.univariate.multiseq.deep import LstmForecast
from ts.plot import Plot


def main():
    n = 20000
    trainN = 19000

    trainData, testData = Utility.trainTestSplit(
        StandardGenerator('simple').generate(n),
        train=trainN
    )

    Plot.plotDataCols(trainData)

    model = LstmForecast(
        forecastHorizon=1,
        stateSize=10,
        activation='tanh',
        numRnnLayers=1
    )

    losses = model.train(
        trainSequences=Utility.breakSeq(trainData, seqLength=200),
        numIterations=20,
        optimizer=tf.keras.optimizers.Adam(0.02),
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