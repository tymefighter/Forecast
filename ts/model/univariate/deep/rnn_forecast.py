import tensorflow as tf
import numpy as np

from ts.utility import Utility
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
        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.MSE)
        callbacks = None
        if modelSavePath is not None:
            callbacks = [tf.keras.callbacks.ModelCheckpoint(
                modelSavePath,
                'train_loss',
                save_freq='epoch'
            )]

        history = self.model.fit(
            GeneratedDataSequence(
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

        assert (Utility.isExoShapeValid(exogenousSeries, self.inputDimension - 1))
        X = Utility.prepareDataPred(targetSeries, exogenousSeries)

        return tf.squeeze(self.model.predict(np.expand_dims(X, axis=0)))

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

        assert (Utility.isExoShapeValid(exogenousSeries, self.inputDimension - 1))
        X, Ytrue = Utility.prepareDataTrain(targetSeries, exogenousSeries, self.forecastHorizon)

        Ypred = tf.squeeze(self.model.predict(np.expand_dims(X, axis=0)))
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

        self.model.save(modelSavePath, save_format='h5')

    def load(
            self,
            modelLoadPath
    ):
        """
        Load the model parameters from the provided path

        :param modelLoadPath: Path from where the parameters are to be loaded
        :return: None
        """

        self.model = tf.keras.models.load_model(modelLoadPath)


class GeneratedDataSequence(tf.keras.utils.Sequence):

    def __init__(self, trainSequences, forecastHorizon, numExoVariables):
        self.trainSequences = trainSequences
        self.forecastHorizon = forecastHorizon
        self.numExoVariables = numExoVariables

    def __len__(self):
        return len(self.trainSequences)

    def __getitem__(self, idx):

        if type(self.trainSequences[idx]) is tuple:
            targetSeries = self.trainSequences[idx][0]
            exogenousSeries = self.trainSequences[idx][1]
        else:
            targetSeries = self.trainSequences[idx]
            exogenousSeries = None

        assert (Utility.isExoShapeValid(exogenousSeries, self.numExoVariables))

        X, Y = Utility.prepareDataTrain(
            targetSeries,
            exogenousSeries,
            self.forecastHorizon
        )

        return X[np.newaxis, :, :], Y[np.newaxis, :]
