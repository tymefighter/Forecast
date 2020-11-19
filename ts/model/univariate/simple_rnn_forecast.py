import tensorflow as tf

from ts.model.univariate.rnn_forecast import RnnForecast


class SimpleRnnForecast(RnnForecast):
    """ Simple RNN forecasting model """

    def __init__(
            self,
            forecastHorizon=1,
            stateSize=10,
            activation='tanh',
            numRnnLayers=1,
            numExoVariables=0,
            modelLoadPath=None
    ):
        """
        Initialize Simple RNN Forecasting model using the given parameters

        :param forecastHorizon: How much further in the future the model has to
        predict the target series variable
        :param stateSize: Size of the state of each Simple RNN layer
        :param activation: Activation function to use
        :param numRnnLayers: Number of GRU layers of the model
        :param numExoVariables: Number of exogenous variables the model takes as input
        :param modelLoadPath: If specified, then all provided parameters are ignored,
        and the model is loaded from the path
        """

        simpleRnnParam = {
            'units': stateSize,
            'activation': activation,
            'return_sequences': True
        }

        super().__init__(
            forecastHorizon,
            tf.keras.layers.SimpleRNN,
            simpleRnnParam,
            numRnnLayers,
            numExoVariables,
            modelLoadPath
        )

    """
    Methods train, predict, evaluate, save and load are inherited from
    RnnForecast class
    """
