import tensorflow as tf

from ts.model.univariate.deep.rnn_forecast import RnnForecast


class LstmForecast(RnnForecast):

    def __init__(
            self,
            forecastHorizon=1,
            stateSize=10,
            numRnnLayers=1,
            numExoVariables=0,
            modelLoadPath=None
    ):
        super().__init__(
            forecastHorizon,
            tf.keras.layers.LSTM,
            stateSize,
            numRnnLayers,
            numExoVariables,
            modelLoadPath
        )
