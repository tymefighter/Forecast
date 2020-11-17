import tensorflow as tf

from ts.log import GlobalLogger
from ts.data.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.univariate.oneseq.deep import ExtremeTime
from ts.plot import Plot


def tryExtremeOnData(
        typeOfData,
        forecastHorizon,
        memorySize,
        windowSize,
        encoderStateSize,
        lstmStateSize,
        seqLength,
        numIterations,
        optimizer
):

    data = StandardGenerator(typeOfData).generate(5000)
    trainData, testData = Utility.trainTestSplit(data, 4800)

    Plot.plotDataCols(trainData)

    model = ExtremeTime(
        forecastHorizon,
        memorySize,
        windowSize,
        encoderStateSize,
        lstmStateSize
    )

    losses = model.train(
        trainData,
        sequenceLength=seqLength,
        numIterations=numIterations,
        optimizer=optimizer,
        verboseLevel=2,
        returnLosses=True
    )

    Plot.plotLoss(losses)

    evalLoss, predTarget = model.evaluate(testData, returnPred=True)
    trueTarget = testData[1:]

    Plot.plotPredTrue(predTarget, trueTarget)

def main():
    GlobalLogger.getLogger().setLevel(2)

    dataTry = 'long_term'

    if dataTry == 'simple':
        tryExtremeOnData(
            typeOfData='simple',
            forecastHorizon=1,
            memorySize=5,
            windowSize=5,
            encoderStateSize=5,
            lstmStateSize=10,
            seqLength=100,
            numIterations=10,
            optimizer=tf.keras.optimizers.Adam(0.3)
        )
    elif dataTry == 'longTerm':
        tryExtremeOnData(
            typeOfData='long_term',
            forecastHorizon=1,
            memorySize=5,
            windowSize=5,
            encoderStateSize=5,
            lstmStateSize=50,
            seqLength=500,
            numIterations=10,
            optimizer=tf.keras.optimizers.Adam(0.7)
        )
    elif dataTry == 'extreme_short':
        tryExtremeOnData(
            typeOfData='extreme_short',
            forecastHorizon=1,
            memorySize=5,
            windowSize=5,
            encoderStateSize=5,
            lstmStateSize=10,
            seqLength=100,
            numIterations=10,
            optimizer=tf.keras.optimizers.Adam(0.3)
        )
    else:
        tryExtremeOnData(
            typeOfData='extreme_long',
            forecastHorizon=1,
            memorySize=5,
            windowSize=5,
            encoderStateSize=5,
            lstmStateSize=10,
            seqLength=100,
            numIterations=10,
            optimizer=tf.keras.optimizers.Adam(0.3)
        )


if __name__ == '__main__':
    main()
