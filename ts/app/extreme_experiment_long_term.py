import numpy as np
import tensorflow as tf
from ts.data.generate.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.special import ExtremeTime, ExtremeTime2
from ts.plot import Plot

PLOT_DIR = '/Users/ahmed/Programming/Project/image'


def evaluateAndPlot(
        model,
        trainData,
        testData,
        numTrainSeqPlot,
        trainPlotSeqLength,
        plotPrefix,
        plotDir
):

    for i in range(numTrainSeqPlot):
        idx = np.random.randint(0, trainData.shape[0] - trainPlotSeqLength)
        seq = trainData[idx:idx + trainPlotSeqLength]

        evalLoss, Ypred = model.evaluate(seq, returnPred=True)
        Ytrue = seq[1:]
        print(f'Train Eval Loss: {evalLoss}')

        plotPath = plotDir + f'/{plotPrefix}_train{i}.png'
        Plot.plotPredTrue(Ypred, Ytrue, 'Train Data', savePath=plotPath, saveOnly=True)

    evalLoss, Ypred = model.evaluate(testData, returnPred=True)
    Ytrue = testData[1:]
    print(f'Test Eval Loss: {evalLoss}')

    plotPath = plotDir + f'/{plotPrefix}_test.png'
    Plot.plotPredTrue(Ypred, Ytrue, 'Test Data', savePath=plotPath, saveOnly=True)


def tryExtreme1(trainData, testData, seqLength, plotPrefix, plotDir):

    model = ExtremeTime(
        forecastHorizon=1,
        memorySize=20,
        windowSize=10,
        encoderStateSize=10,
        lstmStateSize=10
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

    plotPath = plotDir + f'/{plotPrefix}_loss.png'
    Plot.plotLoss(loss, savePath=plotPath, saveOnly=True)

    numTrainSeqPlot = 5
    evaluateAndPlot(
        model,
        trainData,
        testData,
        numTrainSeqPlot,
        seqLength,
        plotPrefix,
        plotDir
    )


def tryExtreme2(trainData, testData, seqLength, plotPrefix, plotDir):

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

    plotPath = plotDir + f'/{plotPrefix}_loss.png'
    Plot.plotLoss(loss, savePath=plotPath, saveOnly=True)

    numTrainSeqPlot = 5
    evaluateAndPlot(
        model,
        trainData,
        testData,
        numTrainSeqPlot,
        seqLength,
        plotPrefix,
        plotDir
    )


def main():
    # The data generator
    dataGenerator = StandardGenerator('long_term')

    # Generated Data
    n = 21500
    trainN = 21000
    trainData, testData = Utility.trainTestSplit(
        dataGenerator.generate(n),
        train=trainN
    )

    # Extreme Model 1
    seqLength = 500
    tryExtreme1(trainData, testData, seqLength, 'extreme1', PLOT_DIR)

    # Extreme Model 2
    seqLength = 500
    tryExtreme1(trainData, testData, seqLength, 'extreme2', PLOT_DIR)


if __name__ == '__main__':
    main()
