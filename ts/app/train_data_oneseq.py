import numpy as np
import tensorflow as tf
from ts.data.generate.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.univariate import LstmForecast
from ts.plot import Plot

PLOT_DIR = '/Users/ahmed/Programming/Project/image'


def tryModelOneSeq(trainSequences, testData, plotPrefix, plotDir):
    model = LstmForecast(
        forecastHorizon=1,
        stateSize=50,
        activation='tanh',
        numRnnLayers=3
    )

    loss = model.train(
        trainSequences=trainSequences,
        numIterations=15,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                0.03,
                20,
                0.96
            )
        )
    )

    plotPath = plotDir + f'/{plotPrefix}_loss.png'
    Plot.plotLoss(loss, savePath=plotPath, saveOnly=True)

    trainPlot = 5
    for i, idx in enumerate(list(np.random.randint(0, len(trainSequences), size=(trainPlot,)))):

        seq = trainSequences[idx]
        evalLoss, Ypred = model.evaluate(seq, returnPred=True)
        Ytrue = seq[1:]
        print(f'Training Eval Loss: {evalLoss}')

        plotPath = plotDir + f'/{plotPrefix}_train{i}.png'
        Plot.plotPredTrue(
            Ypred,
            Ytrue,
            'Train Data',
            savePath=plotPath,
            saveOnly=True
        )

    evalLoss, Ypred = model.evaluate(testData, returnPred=True)
    Ytrue = testData[1:]
    print(f'Test Eval Loss: {evalLoss}')

    plotPath = plotDir + f'/{plotPrefix}_test.png'
    Plot.plotPredTrue(Ypred, Ytrue, 'Test Data', savePath=plotPath, saveOnly=True)


def tryModelMultiSeq(trainSequences, testData, plotPrefix, plotDir):

    model = LstmForecast(
        forecastHorizon=1,
        stateSize=10,
        activation='tanh',
        numRnnLayers=1
    )

    loss = model.train(
        trainSequences=trainSequences,
        numIterations=15,
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                0.08,
                40,
                0.98
            )
        )
    )

    plotPath = plotDir + f'/{plotPrefix}_loss.png'
    Plot.plotLoss(loss, savePath=plotPath, saveOnly=True)

    trainPlot = 5
    for i, idx in enumerate(list(np.random.randint(0, len(trainSequences), size=(trainPlot,)))):
        seq = trainSequences[idx]
        evalLoss, Ypred = model.evaluate(seq, returnPred=True)
        Ytrue = seq[1:]
        print(f'Training Eval Loss: {evalLoss}')

        plotPath = plotDir + f'/{plotPrefix}_train{i}.png'
        Plot.plotPredTrue(
            Ypred,
            Ytrue,
            'Train Data',
            savePath=plotPath,
            saveOnly=True
        )

    evalLoss, Ypred = model.evaluate(testData, returnPred=True)
    Ytrue = testData[1:]
    print(f'Test Eval Loss: {evalLoss}')

    plotPath = plotDir + f'/{plotPrefix}_test.png'
    Plot.plotPredTrue(Ypred, Ytrue, 'Test Data', savePath=plotPath, saveOnly=True)


def main():
    # The data generator
    dataGenerator = StandardGenerator('long_term')

    # Data for multi-sequence methods
    n = 21500
    trainN = 21000
    trainData, testData = Utility.trainTestSplit(
        dataGenerator.generate(n),
        train=trainN
    )

    # Method 1 - train on mutually exclusive sequences
    seqLength = 500
    trainSequences = Utility.breakSeq(trainData, seqLength)
    tryModelOneSeq(trainSequences, testData, 'method1', PLOT_DIR)

    # Method 2 - train on randomly sampled contiguous sequences
    seqLength = 500
    numSeq = 42
    trainSequences = [
        trainData[startIdx: startIdx + seqLength]
        for startIdx in list(np.random.randint(
            0,
            trainN - seqLength,
            size=(numSeq,)
        ))
    ]
    tryModelOneSeq(trainSequences, testData, 'method2', PLOT_DIR)

    # Method 3 - train on the single long sequence
    trainSequences = [trainData]
    tryModelOneSeq(trainSequences, testData, 'method3', PLOT_DIR)

    # Multiple Train Sequences
    seqLength = 500
    numSeq = 42
    trainSequences = Utility.generateMultipleSequence(
        dataGenerator=dataGenerator,
        numSequences=numSeq,
        minSequenceLength=seqLength,
        maxSequenceLength=seqLength
    )
    testData = dataGenerator.generate(seqLength)
    tryModelMultiSeq(trainSequences, testData, 'multiseq', PLOT_DIR)


if __name__ == '__main__':
    main()
