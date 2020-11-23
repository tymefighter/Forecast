import numpy as np
import tensorflow as tf
from ts.data.generate.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.multivariate import DeepNN
from ts.plot import Plot


def main():

    n = 20200
    trainN = 20000
    seqLength = 500

    data = np.expand_dims(StandardGenerator('long_term').generate(n), axis=1)
    trainData, testData = Utility.trainTestSplit(data, trainN)

    trainSequences = Utility.breakTrainSeq(trainData, None, seqLength)

    forecastHorizon = 1
    lag = 30

    model = DeepNN(
        forecastHorizon=forecastHorizon,
        lag=lag,
        numUnitsPerLayer=10,
        numLayers=2,
        numTargetVariables=1,
        numExoVariables=0
    )

    loss = model.train(
        trainSequences=trainSequences,
        numIterations=20,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                0.1,
                25,
                0.97
            )
        ),
        verboseLevel=2,
        returnLosses=True
    )

    Plot.plotLoss(loss)

    evalLoss, Ypred = model.evaluate(
        testData,
        returnPred=True
    )

    Ytrue = testData[lag + forecastHorizon:, :]

    print(f'Eval Loss: {evalLoss}')
    Plot.plotPredTrue(Ypred, Ytrue)


if __name__ == '__main__':
    main()
