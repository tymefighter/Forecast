import numpy as np
import tensorflow as tf
from ts.utility.utility import Utility


class ForecastDataSequence(tf.keras.utils.Sequence):

    def __init__(self, trainSequences, forecastHorizon, numExoVariables):
        self.trainSequences = trainSequences
        self.forecastHorizon = forecastHorizon
        self.numExoVariables = numExoVariables

    def __len__(self):
        return len(self.trainSequences)

    def __getitem__(self, idx):

        if type(self.trainSequences[idx]) is tuple:
            targetSeries = self.trainSequences[idx][0]
            exogenousSeries = self.trainSequences[idx][1]
        else:
            targetSeries = self.trainSequences[idx]
            exogenousSeries = None

        assert (Utility.isExoShapeValid(exogenousSeries, self.numExoVariables))

        X, Y = Utility.prepareDataTrain(
            targetSeries,
            exogenousSeries,
            self.forecastHorizon
        )

        return X[np.newaxis, :, :], Y[np.newaxis, :]
