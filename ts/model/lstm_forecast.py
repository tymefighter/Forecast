import pickle
import tensorflow as tf

from ts.model.rnn_forecast import RnnForecast
from ts.log import GlobalLogger


class LstmForecast(RnnForecast):
    """ LSTM univariate forecasting model """

    @staticmethod
    def load(modelLoadPath):
        """
        Loads the model from the provided filepath

        :param modelLoadPath: path from where to load the model
        :return: model which is loaded from the given path
        """

        model = LstmForecast(loadModel=True)
        GlobalLogger.getLogger().log('Loading Model', 1, LstmForecast.load.__name__)

        with open(modelLoadPath, 'rb') as loadFile:
            loadDict = pickle.load(loadFile)

        model.forecastHorizon = loadDict['forecastHorizon']
        model.layerClass = loadDict['layerClass']
        model.layerParameters = loadDict['layerParameters']
        model.numRnnLayers = loadDict['numRnnLayers']
        model.numTargetVariables = loadDict['numTargetVariables']
        model.numExoVariables = loadDict['numExoVariables']

        model.buildModel()
        model.model.set_weights(loadDict['weights'])

        GlobalLogger.getLogger().log('Loading Complete', 1, LstmForecast.load.__name__)

        return model

    def __init__(
            self,
            forecastHorizon=1,
            stateSize=10,
            activation='tanh',
            numRnnLayers=1,
            numTargetVariables=1,
            numExoVariables=0,
            loadModel=False
    ):
        """
        Initialize LSTM Forecasting model using the given parameters

        :param forecastHorizon: How much further in the future the model has to
        predict the target series variable
        :param stateSize: Size of the state of each LSTM layer
        :param activation: Activation function to use
        :param numRnnLayers: Number of LSTM layers of the model
        :param numTargetVariables: Number of target variables the model takes as input
        :param numExoVariables: Number of exogenous variables the model takes as input
        :param loadModel: True or False - do not use this parameter !,
        this is for internal use only (i.e. it is an implementation detail)
        If True, then object is normally created, else object is created
        without any member values being created. This is used when model
        is created by the static load method
        """

        if loadModel:
            return

        lstmParam = {
            'units': stateSize,
            'activation': activation,
            'return_sequences': True
        }

        super().__init__(
            forecastHorizon,
            tf.keras.layers.LSTM,
            lstmParam,
            numRnnLayers,
            numTargetVariables,
            numExoVariables
        )

    """
    Methods train, predict, evaluate, save and load are inherited from
    RnnForecast class
    """
