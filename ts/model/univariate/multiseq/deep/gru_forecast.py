import tensorflow as tf

from ts.model.univariate.multiseq.deep import RnnForecast


class GruForecast(RnnForecast):
    """ GRU forecasting model """

    def __init__(
            self,
            forecastHorizon=1,
            stateSize=10,
            numRnnLayers=1,
            numExoVariables=0,
            modelLoadPath=None
    ):
        """
        Initialize GRU Forecasting model using the given parameters

        :param forecastHorizon: How much further in the future the model has to
        predict the target series variable
        :param stateSize: Size of the state of each GRU layer
        :param numRnnLayers: Number of GRU layers of the model
        :param numExoVariables: Number of exogenous variables the model takes as input
        :param modelLoadPath: If specified, then all provided parameters are ignored,
        and the model is loaded from the path
        """

        super().__init__(
            forecastHorizon,
            tf.keras.layers.GRU,
            stateSize,
            numRnnLayers,
            numExoVariables,
            modelLoadPath
        )

    """
    Methods train, predict, evaluate, save and load are inherited from
    RnnForecast class
    """
