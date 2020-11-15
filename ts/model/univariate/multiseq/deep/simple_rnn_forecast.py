import tensorflow as tf

from ts.model.univariate.multiseq.deep import RnnForecast


class SimpleRnnForecast(RnnForecast):

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
            tf.keras.layers.SimpleRNN,
            stateSize,
            numRnnLayers,
            numExoVariables,
            modelLoadPath
        )

    """
    Methods train, predict, evaluate, save and load are inherited from
    RnnForecast class
    """
