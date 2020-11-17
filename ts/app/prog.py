import tensorflow as tf

from ts.log import GlobalLogger
from ts.data.univariate.nonexo import StandardGenerator
from ts.utility import Utility
from ts.model.univariate.multiseq.deep import LstmForecast
from ts.plot import Plot


def main():
    data = StandardGenerator('long_term').generate(10500)
    trainData, testData = Utility.trainTestSplit(data, 10200)

    Plot.plotDataCols(trainData)

    model = LstmForecast(1, 10, 2)
    losses = model.train(
        Utility.breakSeq(trainData, 1000),
        10,
        tf.keras.optimizers.Adam(0.7)
    )

    Plot.plotLoss(losses)

    evalLoss, predTarget = model.evaluate(testData, returnPred=True)
    trueTarget = testData[1:]

    Plot.plotPredTrue(predTarget, trueTarget)


if __name__ == '__main__':
    main()
