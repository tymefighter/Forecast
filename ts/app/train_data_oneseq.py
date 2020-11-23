import numpy as np
import tensorflow as tf
from ts.data.generate.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.univariate import LstmForecast
from ts.plot import Plot


def tryModelOneSeq(trainSequences, testData):

    model = LstmForecast(
        forecastHorizon=1,
        stateSize=120,
        activation='tanh',
        numRnnLayers=3
    )

    loss = model.train(
        trainSequences=trainSequences,
        numIterations=20,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                0.04,
                40,
                0.98
            )
        )
    )

    Plot.plotLoss(loss)

    trainPlot = 5
    for idx in list(np.random.randint(0, len(trainSequences), size=(trainPlot,))):

        seq = trainSequences[idx]
        evalLoss, Ypred = model.evaluate(seq, returnPred=True)
        Ytrue = seq[1:]
        print(f'Training Eval Loss: {evalLoss}')
        Plot.plotPredTrue(Ypred, Ytrue, 'Train Data')

    evalLoss, Ypred = model.evaluate(testData, returnPred=True)
    Ytrue = testData[1:]
    print(f'Test Eval Loss: {evalLoss}')
    Plot.plotPredTrue(Ypred, Ytrue, 'Test Data')


def tryModelMultiSeq(trainSequences, testData):

    model = LstmForecast(
        forecastHorizon=1,
        stateSize=30,
        activation='tanh',
        numRnnLayers=2
    )

    loss = model.train(
        trainSequences=trainSequences,
        numIterations=2,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.05
        )
    )

    Plot.plotLoss(loss)

    trainPlot = 5
    for idx in list(np.random.randint(0, len(trainSequences), size=(trainPlot,))):
        seq = trainSequences[idx]
        evalLoss, Ypred = model.evaluate(seq, returnPred=True)
        Ytrue = seq[1:]
        print(f'Training Eval Loss: {evalLoss}')
        Plot.plotPredTrue(Ypred, Ytrue, 'Train Data')

    evalLoss, Ypred = model.evaluate(testData, returnPred=True)
    Ytrue = testData[1:]
    print(f'Test Eval Loss: {evalLoss}')
    Plot.plotPredTrue(Ypred, Ytrue, 'Test Data')


def main():
    # The data generator
    dataGenerator = StandardGenerator('long_term')

    # # Data for multi-sequence methods
    # n = 20500
    # trainN = 20000
    # trainData, testData = Utility.trainTestSplit(
    #     dataGenerator.generate(n),
    #     train=trainN
    # )
    #
    # # Method 1 - train on the single long sequence
    # trainSequences = [trainData]
    # tryModelOneSeq(trainSequences, testData)
    #
    # # Method 2 - train on mutually exclusive sequences
    # seqLength = 500
    # trainSequences = Utility.breakSeq(trainData, seqLength)
    # tryModelOneSeq(trainSequences, testData)
    #
    # # Method 3 - train on randomly sampled contiguous sequences
    # seqLength = 500
    # trainSequences = [
    #     trainData[startIdx: startIdx + seqLength]
    #     for startIdx in list(np.random.randint(
    #         0,
    #         trainN - seqLength
    #     ))
    # ]
    # tryModelOneSeq(trainSequences, testData)

    # Multiple Train Sequences
    seqLength = 500
    numSeq = 40
    trainSequences = Utility.generateMultipleSequence(
        dataGenerator=dataGenerator,
        numSequences=numSeq,
        minSequenceLength=seqLength,
        maxSequenceLength=seqLength
    )
    testData = dataGenerator.generate(seqLength)
    tryModelMultiSeq(trainSequences, testData)


if __name__ == '__main__':
    main()