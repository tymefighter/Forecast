import tensorflow as tf

from ts.model.rnn_forecast import RnnForecast


class GruForecast(RnnForecast):
    """ GRU univariate forecasting model """

    def __init__(
            self,
            forecastHorizon=1,
            stateSize=10,
            activation='tanh',
            numRnnLayers=1,
            numTargetVariables=1,
            numExoVariables=0,
            modelLoadPath=None
    ):
        """
        Initialize GRU Forecasting model using the given parameters

        :param forecastHorizon: How much further in the future the model has to
        predict the target series variable
        :param stateSize: Size of the state of each GRU layer
        :param activation: Activation function to use
        :param numRnnLayers: Number of GRU layers of the model
        :param numTargetVariables: Number of target variables the model takes as input
        :param numExoVariables: Number of exogenous variables the model takes as input
        :param modelLoadPath: If specified, then all provided parameters are ignored,
        and the model is loaded from the path
        """

        gruParam = {
            'units': stateSize,
            'activation': activation,
            'return_sequences': True
        }

        super().__init__(
            forecastHorizon,
            tf.keras.layers.GRU,
            gruParam,
            numRnnLayers,
            numTargetVariables,
            numExoVariables,
            modelLoadPath
        )

    """
    Methods train, predict, evaluate, save and load are inherited from
    RnnForecast class
    """
