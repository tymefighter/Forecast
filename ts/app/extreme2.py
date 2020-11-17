import tensorflow as tf

from ts.log import GlobalLogger
from ts.data.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.univariate.oneseq.deep import ExtremeTime2
from ts.plot import Plot


def tryExtremeOnData2(
        typeOfData,
        forecastHorizon,
        memorySize,
        windowSize,
        embeddingSize,
        contextSize,
        seqLength,
        numIterations,
        optimizer
):

    data = StandardGenerator(typeOfData).generate(1500)
    trainData, testData = Utility.trainTestSplit(data, 1300)

    Plot.plotDataCols(trainData)

    model = ExtremeTime2(
        forecastHorizon,
        memorySize,
        windowSize,
        embeddingSize,
        contextSize
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
        tryExtremeOnData2(
            typeOfData='simple',
            forecastHorizon=1,
            memorySize=5,
            windowSize=5,
            embeddingSize=5,
            contextSize=10,
            seqLength=100,
            numIterations=10,
            optimizer=tf.keras.optimizers.Adam(0.15)
        )
    elif dataTry == 'long_term':
        tryExtremeOnData2(
            typeOfData='long_term',
            forecastHorizon=1,
            memorySize=5,
            windowSize=5,
            embeddingSize=5,
            contextSize=50,
            seqLength=100,
            numIterations=10,
            optimizer=tf.keras.optimizers.Adam(0.20)
        )
    elif dataTry == 'extreme_short':
        tryExtremeOnData2(
            typeOfData='extreme_short',
            forecastHorizon=1,
            memorySize=5,
            windowSize=5,
            embeddingSize=5,
            contextSize=10,
            seqLength=100,
            numIterations=10,
            optimizer=tf.keras.optimizers.Adam(0.1)
        )
    else:
        tryExtremeOnData2(
            typeOfData='extreme_long',
            forecastHorizon=1,
            memorySize=5,
            windowSize=5,
            embeddingSize=5,
            contextSize=10,
            seqLength=100,
            numIterations=10,
            optimizer=tf.keras.optimizers.Adam(0.3)
        )


if __name__ == '__main__':
    main()
