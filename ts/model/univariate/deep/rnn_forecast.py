import pickle
import tensorflow as tf
import numpy as np

from ts.utility import Utility, ForecastDataSequence
from ts.log import GlobalLogger


class RnnForecast:

    def __init__(
            self,
            forecastHorizon=1,
            stateSize=10,
            numRnnLayers=1,
            numExoVariables=0,
            modelLoadPath=None
    ):
        if modelLoadPath:
            self.load(modelLoadPath)
        else:
            GlobalLogger.getLogger().log(
                "Building Model Architecture",
                1,
                self.__init__.__name__
            )

            self.forecastHorizon = forecastHorizon
            self.model = tf.keras.Sequential()

            for i in range(numRnnLayers):
                self.model.add(tf.keras.layers.SimpleRNN(
                    stateSize,
                    return_sequences=True
                ))

            self.model.add(tf.keras.layers.Dense(1, activation=None))
            self.inputDimension = numExoVariables + 1
            self.model.build(input_shape=(None, None, self.inputDimension))

    def train(
            self,
            trainSequences,
            optimizer=tf.optimizers.Adam(),
            modelSavePath=None,
            verboseLevel=1,
            returnLosses=True,
            numIterations=1
    ):
        logger = GlobalLogger.getLogger()
        logger.log("Compiling Model", 1, self.train.__name__)

        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.MSE)

        callbacks = None
        if modelSavePath is not None:
            callbacks = [SaveCallback(
                self,
                modelSavePath
            )]

        logger.log("Begin Training Model", 1, self.train.__name__)
        history = self.model.fit(
            ForecastDataSequence(
                trainSequences,
                self.forecastHorizon,
                self.inputDimension - 1
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

        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (n + self.forecastHorizon,)
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
        assert (Utility.isExoShapeValid(exogenousSeries, self.inputDimension - 1))
        X = Utility.prepareDataPred(targetSeries, exogenousSeries)

        logger.log('Begin Prediction', 1, self.predict.__name__)
        return tf.squeeze(self.model.predict(np.expand_dims(X, axis=0), verbose=0))

    def evaluate(
            self,
            targetSeries,
            exogenousSeries=None,
            returnPred=False
    ):
        """
        Forecast using the model parameters on the provided data, evaluates
        the forecast result using the loss and returns it

        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (numTimesteps + self.forecastHorizon,).
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
        assert (Utility.isExoShapeValid(exogenousSeries, self.inputDimension - 1))
        X, Ytrue = Utility.prepareDataTrain(targetSeries, exogenousSeries, self.forecastHorizon)

        logger.log('Begin Evaluation', 1, self.predict.__name__)
        Ypred = tf.squeeze(self.model.predict(np.expand_dims(X, axis=0), verbose=0))
        loss = tf.keras.losses.MSE(
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
        self.model.save(
            modelSavePath,
            include_optimizer=False,
            save_format='tf'
        )

        modelSavePath += '/info'
        fl = open(modelSavePath, 'wb')
        saveDict = {
            'forecastHorizon': self.forecastHorizon,
            'inputDimension': self.inputDimension
        }
        pickle.dump(saveDict, fl)
        fl.close()

    def load(
            self,
            modelLoadPath
    ):
        """
        Load the model parameters from the provided path

        :param modelLoadPath: Path from where the parameters are to be loaded
        :return: None
        """

        GlobalLogger.getLogger().log('Loading Model', 1, self.load.__name__)
        self.model = tf.keras.models.load_model(
            modelLoadPath,
            compile=False
        )

        modelLoadPath += '/info'
        fl = open(modelLoadPath, 'rb')
        loadDict = pickle.load(fl)
        fl.close()

        self.forecastHorizon = loadDict['forecastHorizon']
        self.inputDimension = loadDict['inputDimension']


class SaveCallback(tf.keras.callbacks.Callback):

    def __init__(self, rnnForecastModel, modelSavePath):
        super().__init__()
        self.rnnForecastModel = rnnForecastModel
        self.modelSavePath = modelSavePath

    def on_epoch_end(self, epoch, logs=None):
        self.rnnForecastModel.save(self.modelSavePath)
