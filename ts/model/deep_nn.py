import pickle
import numpy as np
import tensorflow as tf

from ts.utility import Utility, SaveCallback
from ts.log import GlobalLogger


class DeepNN:
    """ Deep Neural Network based forecasting model """

    def __init__(
            self,
            forecastHorizon=1,
            lag=1,
            activation='relu',
            numUnitsPerLayer=10,
            numLayers=1,
            numTargetVariables=1,
            numExoVariables=0,
            modelLoadPath=None
    ):
        """

        :param forecastHorizon: How much further in the future the model has to
        predict the target series variable
        :param lag: The lag to be considered
        :param activation: The activation function of each layer in the network
        :param numUnitsPerLayer: The number of units per layer
        :param numLayers: Number of layers
        :param numTargetVariables: Number of target variables
        :param numExoVariables: Number of exogenous variables
        :param modelLoadPath: If specified, then all provided parameters are ignored,
        and the model is loaded from the path
        """

        if modelLoadPath is not None:
            self.load(modelLoadPath)
        else:
            self.forecastHorizon = forecastHorizon
            self.lag = lag
            self.activation = activation
            self.numUnitsPerLayer = numUnitsPerLayer
            self.numLayers = numLayers
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

        :param trainSequences: Sequences (List) of data, each element in the
        list is a target sequence of shape (n1, numTargetVariables) or a tuple
        containing a target sequence of shape (n1 + forecastHorizon, numTargetVariables)
        and an exogenous sequence of shape (n1, numExoVariables)
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

        self.model.compile(optimizer=optimizer, loss=tf.keras.losses.MSE)

        callbacks = None
        if modelSavePath is not None:
            callbacks = [SaveCallback(
                self,
                modelSavePath
            )]

        logger.log('Begin Training Model', 1, self.train.__name__)
        history = self.model.fit(
            DnnDataSequence(
                trainSequences,
                self.forecastHorizon,
                self.numTargetVariables,
                self.numExoVariables,
                self.lag
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

        V.imp: Predicted value is NOT outputted for the first lag - 1 inputs,
        since then networks begins predicting from the lag term onwards

        :param targetSeries: Multivariate Series of the Target Variable, it
        should be a numpy array of shape (lag - 1 + nPred, numTargetVariables)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (lag - 1 + nPred, numExoVariables), it can be None only if
        numExoVariables is 0 in which case the exogenous variables are not
        considered
        :return: Forecast targets predicted by the model, it has shape
        (nPred, numTargetVariables), the horizon of the targets is the
        same as self.forecastHorizon
        """

        logger = GlobalLogger.getLogger()

        logger.log(f'Target Series Shape: {targetSeries.shape}', 2, self.predict.__name__)
        if exogenousSeries is not None:
            logger.log(
                f'Exogenous Series Shape: {exogenousSeries.shape}', 2, self.predict.__name__
            )

        logger.log('Prepare Data', 1, self.predict.__name__)
        assert (self.checkShapeValid(targetSeries, exogenousSeries))

        X = DeepNN.prepareDataPredDNN(targetSeries, exogenousSeries, self.lag)

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

        V.imp: Predicted value is NOT outputted for the first lag - 1 inputs,
        since then networks begins predicting from the lag term onwards. Also
        the last forecastHorizon terms are not taken as the input, they are
        taken as part of the true output which would be used for evalutation

        :param targetSeries: Multivariate Series of the Target Variable, it
        should be a numpy array of shape
        (self.lag - 1 + numTimesteps + self.forecastHorizon, numTargetVariables).
        numTimesteps is the number of timesteps on which our model must predict,
        the values ahead are for evaluating the predicted results with respect
        to them (i.e. they are true targets for our prediction)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (self.lag - 1 + numTimesteps, numExoVariables), it can be None
        only if numExoVariables is 0 in which case the exogenous variables
        are not considered
        :param returnPred: If True, then return predictions along with loss, else
        return on loss
        :return: If True, then return predictions along with loss of the predicted
        and true targets, else return only loss. The predictions would have shape
        (numTimesteps, numTargetVariables)
        """

        logger = GlobalLogger.getLogger()

        logger.log(f'Target Series Shape: {targetSeries.shape}', 2, self.evaluate.__name__)
        if exogenousSeries is not None:
            logger.log(
                f'Exogenous Series Shape: {exogenousSeries.shape}', 2, self.evaluate.__name__
            )

        logger.log('Prepare Data', 1, self.evaluate.__name__)

        assert (self.checkShapeValid(targetSeries, exogenousSeries))

        X, Ytrue = DeepNN.prepareDataTrainDNN(
            targetSeries, exogenousSeries, self.forecastHorizon, self.lag
        )

        logger.log('Begin Evaluation', 1, self.predict.__name__)
        Ypred = self.model.predict(X, verbose=0)

        assert (Ytrue.shape == Ypred.shape)

        loss = DeepNN.lossFunc(Ytrue, Ypred)

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
            'lag': self.lag,
            'activation': self.activation,
            'numUnitsPerLayer': self.numUnitsPerLayer,
            'numLayers': self.numLayers,
            'numTargetVariables': self.numTargetVariables,
            'numExoVariables': self.numExoVariables,
            'weights': self.model.get_weights()
        }

        saveFile = open(modelSavePath, 'wb')
        pickle.dump(saveDict, saveFile)
        saveFile.close()

        GlobalLogger.getLogger().log('Saving Complete', 1, self.save.__name__)

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

        loadFile = open(modelLoadPath, 'rb')
        loadDict = pickle.load(loadFile)
        loadFile.close()

        self.forecastHorizon = loadDict['forecastHorizon']
        self.lag = loadDict['lag']
        self.activation = loadDict['activation']
        self.numUnitsPerLayer = loadDict['numUnitsPerLayer']
        self.numLayers = loadDict['numLayers']
        self.numTargetVariables = loadDict['numTargetVariables']
        self.numExoVariables = loadDict['numExoVariables']

        self.buildModel()
        self.model.set_weights(loadDict['weights'])

        GlobalLogger.getLogger().log('Loading Complete', 1, self.load.__name__)

    def buildModel(self):
        """ Builds Model Architecture """

        GlobalLogger.getLogger().log(
            'Building Model Architecture',
            1,
            self.buildModel.__name__
        )

        inputDimension = (self.numExoVariables + self.numTargetVariables) * (self.lag + 1)

        self.model = tf.keras.Sequential()
        for i in range(self.numLayers):
            self.model.add(tf.keras.layers.Dense(
                units=self.numUnitsPerLayer,
                activation=self.activation
            ))

        self.model.add(tf.keras.layers.Dense(
            units=self.numTargetVariables,
            activation=None
        ))

        self.model.build(input_shape=(None, inputDimension))

    def checkShapeValid(self, targetSeries, exogenousSeries):
        """
        Checks if shape of the target series and exogenous series
        is valid

        :param targetSeries: The target series
        :param exogenousSeries: The exogenous series
        :return: returns True if target series has a shape (n1, numTargetVariables)
        and exogenous series has a shape (n2, numExoVariables) if it is not None,
        it can be none only when numExoVariables is 0. If any of these is not satisfied,
        then False is returned
        """

        return len(targetSeries.shape) == 2 and \
            targetSeries.shape[1] == self.numTargetVariables and \
            Utility.isExoShapeValid(exogenousSeries, self.numExoVariables)

    @staticmethod
    def prepareDataPredDNN(
            targetSeries,
            exogenousSeries,
            lag
    ):
        """
        Prepare Data For Prediction

        :param targetSeries: Multivariate Series of the Target Variable, it
        should be a numpy array of shape (lag - 1 + nPred, numTargetVariables)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (lag - 1 + nPred, numExoVariables), it can be None only if
        numExoVariables is 0 in which case the exogenous variables are not
        considered
        :param lag: The lag to be considered
        :return: Prepared Feature Data X of shape (nPred, numTargetVariables + numExoVariables)
        """

        Xtemp = Utility.prepareDataPred(
            targetSeries,
            exogenousSeries
        )

        X = []

        for i in range(lag, Xtemp.shape[0]):
            vecLen = (lag + 1) * Xtemp.shape[1]
            vec = np.reshape(Xtemp[i - lag: i + 1, :], (vecLen,))
            X.append(vec)

        X = np.array(X)
        return X

    @staticmethod
    def prepareDataTrainDNN(
            targetSeries,
            exogenousSeries,
            forecastHorizon,
            lag
    ):
        """
        Prepare Data For Training

        :param targetSeries: Multivariate Series of the Target Variable, it
        should be a numpy array of shape (lag - 1 + nTrain + forecastHorizon, numTargetVariables)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (lag - 1 + nTrain, numTargetVariables),
        it can be None only if numExoVariables is 0 in which case the exogenous variables are not
        considered
        :param forecastHorizon: How much further in the future the model has to
        predict the target series variable
        :param lag: The lag to be considered
        :return: Prepared training data X of shape (nTrain, numTargetVariables + numExoVariables),
        Y of shape (nTrain, numTargetVariables)
        """

        Xtemp, Ytemp = Utility.prepareDataTrain(
            targetSeries,
            exogenousSeries,
            forecastHorizon
        )

        X = []
        Y = Ytemp[lag:]

        for i in range(lag, Xtemp.shape[0]):
            vecLen = (lag + 1) * Xtemp.shape[1]
            vec = np.reshape(Xtemp[i - lag: i + 1, :], (vecLen,))
            X.append(vec)

        X = np.array(X)
        return X, Y

    @staticmethod
    def lossFunc(Ytrue, Ypred):
        """
        The Loss Function - Mean Square Loss Between every true value
        and predicted value

        :param Ytrue: True targets, it has shape (n, numTargetVariables)
        :param Ypred: Predicted targets, it has shape (n, numTargetVariables)
        :return: The mean square error between the true targets and predicted
        target values
        """

        return tf.reduce_mean(tf.math.square(Ytrue - Ypred))


class DnnDataSequence(tf.keras.utils.Sequence):
    """ Deep NN Data Sequence Provider """

    def __init__(
            self,
            trainSequences,
            forecastHorizon,
            numTargetVariables,
            numExoVariables,
            lag
    ):
        """

        :param trainSequences: Training Sequences
        :param forecastHorizon: The forecast horizon
        :param numTargetVariables: Number of target variables
        :param numExoVariables: Number of exogenous variables
        :param lag: The lag to be considered
        """

        self.trainSequences = trainSequences
        self.forecastHorizon = forecastHorizon
        self.numTargetVariables = numTargetVariables
        self.numExoVariables = numExoVariables
        self.lag = lag

    def __len__(self):
        """ Gets the Number of Batches """

        return len(self.trainSequences)

    def __getitem__(self, idx):
        """
        Gets the batch corresponding to the provided index

        :param idx: Index of the batch which is requested
        :return: The (idx)th batch
        """

        if type(self.trainSequences[idx]) is tuple:
            targetSeries = self.trainSequences[idx][0]
            exogenousSeries = self.trainSequences[idx][1]
        else:
            targetSeries = self.trainSequences[idx]
            exogenousSeries = None

        assert (
                len(targetSeries.shape) == 2
                and
                targetSeries.shape[1] == self.numTargetVariables
        )
        assert (Utility.isExoShapeValid(exogenousSeries, self.numExoVariables))

        X, Y = DeepNN.prepareDataTrainDNN(
            targetSeries,
            exogenousSeries,
            self.forecastHorizon,
            self.lag
        )

        return X, Y
