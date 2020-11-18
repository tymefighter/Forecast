import numpy as np
import tensorflow as tf
from ts.data.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.univariate.multiseq.deep import LstmForecast
from ts.plot import Plot


def main():
    NUM_TRAIN_SEQ = 100
    MIN_SEQ_LENGTH = 100
    MAX_SEQ_LENGTH = 200

    dataGen = StandardGenerator('long_term')
    trainData = [
        dataGen.generate(n)
        for n in list(
            np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH + 1, size=(NUM_TRAIN_SEQ,))
        )
    ]

    testData = dataGen.generate(np.random.randint(MIN_SEQ_LENGTH, MAX_SEQ_LENGTH))

    Plot.plotDataCols(trainData[np.random.randint(0, NUM_TRAIN_SEQ)])

    model = LstmForecast(
        forecastHorizon=1,
        stateSize=10,
        activation='tanh',
        numRnnLayers=1
    )

    losses = model.train(
        trainSequences=trainData,
        numIterations=20,
        optimizer=tf.keras.optimizers.Adam(0.01),
        verboseLevel=2,
        returnLosses=True
    )

    Plot.plotLoss(losses)

    idx = np.random.randint(0, NUM_TRAIN_SEQ)
    evalLoss, Ypred = model.evaluate(trainData[idx], returnPred=True)
    Ytrue = trainData[idx][1:]
    print(f'Training Eval Loss: {evalLoss}')
    Plot.plotPredTrue(Ypred, Ytrue)

    evalLoss, Ypred = model.evaluate(testData, returnPred=True)
    Ytrue = testData[1:]
    print(f'Test Eval Loss: {evalLoss}')
    Plot.plotPredTrue(Ypred, Ytrue)


if __name__ == '__main__':
    main()