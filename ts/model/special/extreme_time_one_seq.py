import pickle
import time

import tensorflow as tf
import numpy as np

from ts.utility import Utility
from ts.log import GlobalLogger, ConsoleLogger


class ExtremeTimeOneSeq:
    """
    Class implementing modified version of the following paper to
    run on one large sequence,

    Name: "Modeling Extreme Events in Time Series Prediction"
    Link: http://staff.ustc.edu.cn/~hexn/papers/kdd19-timeseries.pdf
    """

    def __init__(
            self
    ):
        pass

    def train(
            self,
            targetSeries,
            sequenceLength,
            exogenousSeries=None,
            numIterations=1,
            optimizer=tf.optimizers.Adam(),
            modelSavePath=None,
            verboseLevel=1,
            returnLosses=True
    ):
        pass

    def predict(
            self,
            targetSeries,
            exogenousSeries=None,
    ):
        pass

    def save(
            self,
            modelSavePath
    ):
        """
        Save the model parameters at the provided path

        :param modelSavePath: Path where the parameters are to be saved
        :return: None
        """

        pass

    def load(
            self,
            modelLoadPath
    ):
        """
        Load the model parameters from the provided path

        :param modelLoadPath: Path from where the parameters are to be loaded
        :return: None
        """

        pass
