import numpy as np
import tensorflow as tf
from ts.data.generate.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.univariate import LstmForecast
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

    trainSequences = Utility.breakSeq(trainData, seqLength=seqLength)

    # for i in range(numSeqPlot):
    #     Plot.plotDataCols(trainSequences[
    #         np.random.randint(0, len(trainSequences))
    #     ])

    model = LstmForecast(
        forecastHorizon=1,
        stateSize=20,
        activation='tanh',
        numRnnLayers=2
    )

    model.model.summary()

    loss = model.train(
        trainSequences=trainSequences,
        numIterations=35,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                0.07,
                15,
                0.97
            )
        )
    )

    Plot.plotLoss(loss)

    for i in range(numSeqPlot):
        idx = np.random.randint(0, len(trainSequences))
        evalLoss, Ypred = model.evaluate(trainSequences[idx], returnPred=True)
        Ytrue = trainSequences[idx][1:]

        Plot.plotPredTrue(Ypred, Ytrue, 'On Train')

    evalLoss, Ypred = model.evaluate(testData, returnPred=True)
    Ytrue = testData[1:]

    Plot.plotPredTrue(Ypred, Ytrue, 'On Test')


if __name__ == '__main__':
    main()
