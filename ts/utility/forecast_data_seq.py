import numpy as np
import tensorflow as tf
from ts.utility.utility import Utility


class ForecastDataSequence(tf.keras.utils.Sequence):
    """
    Encapsulates training sequences (data) in a form accepted by tensorflow
    based recurrent models for training
    """

    def __init__(
            self,
            trainSequences,
            forecastHorizon,
            numTargetVariables,
            numExoVariables
    ):
        """
        Create ForecastDataSequence instance using the provided data

        :param trainSequences: Sequences (List) of data, each element in the
        list is a target sequence of shape (n, numTargetVariables) or a tuple
        containing a target sequence of shape (n + forecastHorizon, numTargetVariables)
        and an exogenous sequence of shape (n, numExoVariables)
        :param forecastHorizon: How much further in the future the model has to
        predict the target series variable
        :param numTargetVariables: Number of target variables the model takes as input
        :param numExoVariables: Number of exogenous variables the model takes as input
        """

        self.trainSequences = trainSequences
        self.forecastHorizon = forecastHorizon
        self.numTargetVariables = numTargetVariables
        self.numExoVariables = numExoVariables

    def __len__(self):
        """ returns the number of training sequences """

        return len(self.trainSequences)

    def __getitem__(self, idx):
        """ Get the 'idx' th training sequence as an input-output 2-tuple """

        if type(self.trainSequences[idx]) is tuple:
            targetSeries = self.trainSequences[idx][0]
            exogenousSeries = self.trainSequences[idx][1]
        else:
            targetSeries = self.trainSequences[idx]
            exogenousSeries = None

        assert targetSeries.shape[1] == self.numTargetVariables
        assert Utility.isExoShapeValid(exogenousSeries, self.numExoVariables)

        X, Y = Utility.prepareDataTrain(
            targetSeries,
            exogenousSeries,
            self.forecastHorizon
        )

        return X[np.newaxis, :, :], Y[np.newaxis, :, :]
