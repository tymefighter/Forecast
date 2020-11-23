import pickle
import numpy as np
import tensorflow as tf

from ts.utility import Utility, SaveCallback
from ts.log import GlobalLogger


class DeepNN:

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
        return len(targetSeries.shape) == 2 and \
            targetSeries.shape[1] == self.numTargetVariables and \
            Utility.isExoShapeValid(exogenousSeries, self.numExoVariables)

    @staticmethod
    def prepareDataPredDNN(
            targetSeries,
            exogenousSeries,
            lag
    ):

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
        return tf.reduce_mean(tf.math.square(Ytrue - Ypred))


class DnnDataSequence(tf.keras.utils.Sequence):

    def __init__(
            self,
            trainSequences,
            forecastHorizon,
            numTargetVariables,
            numExoVariables,
            lag
    ):
        self.trainSequences = trainSequences
        self.forecastHorizon = forecastHorizon
        self.numTargetVariables = numTargetVariables
        self.numExoVariables = numExoVariables
        self.lag = lag

    def __len__(self):
        return len(self.trainSequences)

    def __getitem__(self, idx):

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
