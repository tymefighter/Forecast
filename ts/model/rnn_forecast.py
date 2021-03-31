import pickle
import tensorflow as tf
import numpy as np

from ts.utility import Utility, ForecastDataSequence, SaveCallback
from ts.log import GlobalLogger


class RnnForecast:
    """
    RNN based forecasting model which allows for any layer to
    be provided as input
    """

    @staticmethod
    def load(modelLoadPath):
        """
        Loads the model from the provided filepath

        :param modelLoadPath: path from where to load the model
        :return: model which is loaded from the given path
        """

        model = RnnForecast(None, None, None, loadModel=True)
        GlobalLogger.getLogger().log('Loading Model', 1, RnnForecast.load.__name__)

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

        GlobalLogger.getLogger().log('Loading Complete', 1, RnnForecast.load.__name__)

        return model

    def __init__(
            self,
            forecastHorizon,
            layerClass,
            layerParameters,
            numRnnLayers=1,
            numTargetVariables=1,
            numExoVariables=0,
            loadModel=False
    ):
        """
        Initialize RNN-based Forecasting model using the given parameters

        :param forecastHorizon: How much further in the future the model has to
        predict the target series variable
        :param layerClass: Class of the layers of the model
        :param layerParameters: Parameters of the layer passed as a dictionary
        :param numRnnLayers: Number of RNN based layers of the model
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

        self.forecastHorizon = forecastHorizon
        self.layerClass = layerClass
        self.layerParameters = layerParameters
        self.numRnnLayers = numRnnLayers
        self.numTargetVariables = numTargetVariables
        self.numExoVariables = numExoVariables
        self.model = None

        self.buildModel()

    def train(
            self,
            trainSequences,
            numIterations=1,
            optimizer=tf.optimizers.Adam(),
            modelSavePath=None,
            verboseLevel=1,
            returnLosses=True
    ):
        """
        Train the model on the provided data sequences

        :param trainSequences: Sequences of data, each seq in this must either
        be a numpy array of shape (n + forecastHorizon, d1) or a 2-tuple whose
        first element is a numpy array of shape (n + forecastHorizon, d1),
        and second element is a numpy array of shape (n + forecastHorizon, d2)
        :param numIterations: Number of iterations of training to be performed
        :param optimizer: Optimizer using which to train the parameters of the model
        :param modelSavePath: If not None, then save the model to this path after
        every iteration of training
        :param verboseLevel: Verbosity Level, higher value means more information
        :param returnLosses: If True, then return losses of every iteration, else
        does not return losses
        :return: If returnLosses is True, then return list of losses of every
        iteration, else None
        """

        logger = GlobalLogger.getLogger()
        logger.log('Compiling Model', 1, self.train.__name__)

        self.model.compile(
            optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError()
        )

        callbacks = None
        if modelSavePath is not None:
            callbacks = [SaveCallback(
                self,
                modelSavePath
            )]

        logger.log('Begin Training Model', 1, self.train.__name__)
        history = self.model.fit(
            ForecastDataSequence(
                trainSequences,
                self.forecastHorizon,
                self.numTargetVariables,
                self.numExoVariables
            ),
            epochs=numIterations,
            verbose=verboseLevel,
            callbacks=callbacks
        )

        if returnLosses:
            return history.history['loss']

    def predict(
            self,
            targetSeries,
            exogenousSeries=None,
    ):
        """
        Forecast using the model parameters on the provided input data

        :param targetSeries: Series of the Target Variable, it
        should be a numpy array of shape (n, numTargetVariables)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (n, numExoVariables), it can be None only if
        numExoVariables is 0 in which case the exogenous variables are not
        considered
        :return: Forecast targets predicted by the model, it has shape (n,), the
        horizon of the targets is the same as self.forecastHorizon
        """

        logger = GlobalLogger.getLogger()

        logger.log(f'Target Series Shape: {targetSeries.shape}', 2, self.predict.__name__)
        if exogenousSeries is not None:
            logger.log(
                f'Exogenous Series Shape: {exogenousSeries.shape}', 2, self.predict.__name__
            )

        logger.log('Prepare Data', 1, self.predict.__name__)

        assert targetSeries.shape[1] == self.numTargetVariables
        assert (Utility.isExoShapeValid(exogenousSeries, self.numExoVariables))

        X = Utility.prepareDataPred(targetSeries, exogenousSeries)

        logger.log('Begin Prediction', 1, self.predict.__name__)
        return tf.squeeze(self.model.predict(np.expand_dims(X, axis=0), verbose=0), axis=0)

    def evaluate(
            self,
            targetSeries,
            exogenousSeries=None,
            returnPred=False
    ):
        """
        Forecast using the model parameters on the provided data, evaluates
        the forecast result using the loss and returns it

        :param targetSeries: Series of the Target Variable, it
        should be a numpy array of shape
        (numTimesteps + self.forecastHorizon, numTargetVariables).
        numTimesteps is the number of timesteps on which our model must predict,
        the values ahead are for evaluating the predicted results with respect
        to them (i.e. they are true targets for our prediction)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (numTimesteps, numExoVariables), it can be None
        only if numExoVariables is 0 in which case the exogenous variables
        are not considered
        :param returnPred: If True, then return predictions along with loss, else
        return on loss
        :return: If True, then return predictions along with loss of the predicted
        and true targets, else return only loss
        """

        logger = GlobalLogger.getLogger()

        logger.log(f'Target Series Shape: {targetSeries.shape}', 2, self.evaluate.__name__)
        if exogenousSeries is not None:
            logger.log(
                f'Exogenous Series Shape: {exogenousSeries.shape}', 2, self.evaluate.__name__
            )

        logger.log('Prepare Data', 1, self.evaluate.__name__)

        assert targetSeries.shape[1] == self.numTargetVariables
        assert Utility.isExoShapeValid(exogenousSeries, self.numExoVariables)

        X, Ytrue = Utility.prepareDataTrain(targetSeries, exogenousSeries, self.forecastHorizon)

        logger.log('Begin Evaluation', 1, self.predict.__name__)
        Ypred = tf.squeeze(self.model.predict(np.expand_dims(X, axis=0), verbose=0), axis=0)
        loss = tf.keras.losses.MeanSquaredError()(
            Ytrue,
            Ypred
        )

        if returnPred:
            return loss, Ypred
        else:
            return loss

    def save(
            self,
            modelSavePath
    ):
        """
        Save the model parameters at the provided path

        :param modelSavePath: Path where the parameters are to be saved
        :return: None
        """

        GlobalLogger.getLogger().log('Saving Model', 1, self.save.__name__)

        saveDict = {
            'forecastHorizon': self.forecastHorizon,
            'layerClass': self.layerClass,
            'layerParameters': self.layerParameters,
            'numRnnLayers': self.numRnnLayers,
            'numTargetVariables': self.numTargetVariables,
            'numExoVariables': self.numExoVariables,
            'weights': self.model.get_weights()
        }

        saveFile = open(modelSavePath, 'wb')
        pickle.dump(saveDict, saveFile)
        saveFile.close()

        GlobalLogger.getLogger().log('Saving Complete', 1, self.save.__name__)

    def buildModel(self):
        """ Builds Model Architecture """

        GlobalLogger.getLogger().log(
            'Building Model Architecture',
            1,
            self.buildModel.__name__
        )

        self.model = tf.keras.Sequential()

        for i in range(self.numRnnLayers):
            self.model.add(self.layerClass(**self.layerParameters))

        self.model.add(tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.numTargetVariables, activation=None)
        ))

        inputDimension = self.numTargetVariables + self.numExoVariables
        self.model.build(input_shape=(None, None, inputDimension))
