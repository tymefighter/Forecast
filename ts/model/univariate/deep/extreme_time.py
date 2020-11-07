import pickle
import time

import tensorflow as tf
import numpy as np

from ts.model.univariate.univariate import UnivariateModel
from ts.log import DEFAULT_LOG_PATH, FileLogger, ConsoleLogger


class ExtremeTime(UnivariateModel):

    def __init__(
            self,
            forecastHorizon=1,
            memorySize=80,
            windowSize=60,
            encoderStateSize=10,
            lstmStateSize=10,
            numExoVariables=0,
            modelLoadPath=None,
            logPath=DEFAULT_LOG_PATH,
            logLevel=1
    ):
        """
        Initialize the model parameters and hyperparameters

        :param forecastHorizon: How much further in the future the model has to
        predict the target series variable
        :param memorySize: Size of the explicit memory unit used by the model, it
        should be a scalar value
        :param windowSize: Size of each window which is to be compressed and stored
        as a memory cell
        :param encoderStateSize: Size of the hidden state of the GRU encoder
        :param lstmStateSize: Size of the hidden state of the LSTM used in the model
        :param numExoVariables: Number of exogenous variables to be used for training
        :param modelLoadPath: If specified, then all provided parameters (except logging)
        are ignored, and the model is loaded from the path
        :param logPath: Path where to log the information
        :param logLevel: Logging level, 0 means no logging, greater values indicate
        more information
        """

        if modelLoadPath is not None:
            self.load(modelLoadPath, logPath, logLevel)
        else:
            logger = FileLogger(logPath, logLevel)
            logger.write('Initializing Members', 1, self.__init__.__name__)

            self.forecastHorizon = forecastHorizon
            self.memorySize = memorySize
            self.windowSize = windowSize
            self.encoderStateSize = encoderStateSize
            self.lstmStateSize = lstmStateSize
            self.inputDimension = numExoVariables + 1
            self.memory = None
            self.q = None

            logger.write('Building Model Parameters', 1, self.__init__.__name__)

            self.gruEncoder = tf.keras.layers.GRUCell(self.encoderStateSize)
            self.gruEncoder.build(input_shape=(self.inputDimension,))

            self.lstm = tf.keras.layers.LSTMCell(self.lstmStateSize)
            self.lstm.build(input_shape=(self.inputDimension,))

            self.W = tf.Variable(tf.random.normal((1, self.lstmStateSize)))
            self.A = tf.Variable(
                tf.random.normal((self.encoderStateSize, self.lstmStateSize))
            )

            self.b = tf.Variable(0)

            logger.close()

    def train(
            self,
            targetSeries,
            sequenceLength,
            exogenousSeries=None,
            optimizer=tf.optimizers.Adam,
            modelSavePath=None,
            verboseLevel=1,
            logPath=DEFAULT_LOG_PATH,
            logLevel=1
    ):
        """
        Train the Model Parameters on the provided data

        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (n + self.forecastHorizon,)
        :param sequenceLength: Length of each training sequence
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (n, self.numExoVariables), it can be None only if
        self.numExoVariables is 0 in which case the exogenous variables are not
        considered
        :param optimizer: Optimizer of training the parameters
        :param modelSavePath: Path where to save the model parameters after
        each training an a sequence, if None then parameters are not saved
        :param verboseLevel: Verbose level, 0 is nothing, greater values increases
        the information printed to the console
        :param logPath: Path where to log the information
        :param logLevel: Logging level, 0 means no logging, greater values indicate
        more information
        :return: None
        """

        logger = FileLogger(logPath, logLevel)
        verbose = ConsoleLogger(verboseLevel)

        X, Y = self.prepareData(targetSeries, exogenousSeries, logger)

        seqStartTime = self.windowSize
        n = X.shape[0]
        logger.write(
            f'Seq Start Time: {seqStartTime}, Train len: {n}', 2, self.train.__name__
        )
        assert (seqStartTime < n)

        logger.write('Begin Training', 1, self.train.__name__)
        while seqStartTime < n:
            seqEndTime = min(seqStartTime + sequenceLength, n - 1)

            startTime = time.time()
            loss = self.trainSequence(X, Y, seqStartTime, seqEndTime, optimizer, logger)
            endTime = time.time()
            timeTaken = endTime - startTime

            verbose.write(
                f'start timestep: {seqStartTime}'
                + f' | end timestep: {seqEndTime}'
                + f' | time taken: {timeTaken : .2f} sec'
                + f' | Loss: {loss}',
                1
            )

            if modelSavePath is not None:
                logger.write(f'Saving Model at {modelSavePath}', 1, self.train.__name__)

        logger.close()

    def predict(
            self,
            targetSeries,
            exogenousSeries=None,
            logPath=DEFAULT_LOG_PATH,
            logLevel=1
    ):
        """
        Forecast using the model parameters on the provided input data

        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (n,)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (n, self.numExoVariables), it can be None only if
        self.numExoVariables is 0 in which case the exogenous variables are not
        considered
        :param logPath: Path where to log the information
        :param logLevel: Logging level, 0 means no logging, greater values indicate
        more information
        :return: Forecast targets predicted by the model, it has shape (n,), the
        horizon of the targets is the same as self.forecastHorizon
        """
        pass

    def evaluate(
            self,
            targetSeries,
            exogenousSeries=None,
            logPath=DEFAULT_LOG_PATH,
            logLevel=1
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
        numpy array of shape (numTimesteps, self.numExoVariables), it can be None
        only if self.numExoVariables is 0 in which case the exogenous variables
        are not considered
        :param logPath: Path where to log the information
        :param logLevel: Logging level, 0 means no logging, greater values indicate
        more information
        :return: Loss of the predicted and true targets
        """
        pass

    def save(
            self,
            modelSavePath,
            logPath=DEFAULT_LOG_PATH,
            logLevel=1
    ):
        """
        Save the model parameters at the provided path

        :param modelSavePath: Path where the parameters are to be saved
        :param logPath: Path where to log the information
        :param logLevel: Logging level, 0 means no logging, greater values indicate
        more information
        :return: None
        """

        logger = FileLogger(logPath, logLevel)

        assert(self.memory is not None)
        logger.write(f'Memory Shape: {self.memory.shape}', 2, self.save.__name__)

        logger.write('Constructing Dictionary from model params', 1, self.save.__name__)

        saveDict = {
            'memorySize': self.memorySize,
            'windowSize': self.windowSize,
            'inputDimension': self.inputDimension,
            'encoderStateSize': self.encoderStateSize,
            'lstmStateSize': self.lstmStateSize,
            'memory': self.memory,
            'q': self.q,
            'gruEncoder': self.gruEncoder.get_weights(),
            'lstm': self.lstm.get_weights(),
            'W': self.W.read_value(),
            'A': self.A.read_value(),
            'b': self.b.read_value()
        }

        logger.write('Saving Dictionary', 1, self.save.__name__)

        fl = open(modelSavePath, 'wb')
        pickle.dump(saveDict, fl)
        fl.close()

        logger.close()

    def load(
            self,
            modelLoadPath,
            logPath=DEFAULT_LOG_PATH,
            logLevel=1
    ):
        """
        Load the model parameters from the provided path

        :param modelLoadPath: Path from where the parameters are to be loaded
        :param logPath: Path where to log the information
        :param logLevel: Logging level, 0 means no logging, greater values indicate
        more information
        :return: None
        """

        logger = FileLogger(logPath, logLevel)
        logger.write('Load Dictionary from Model Params file', 1, self.load.__name__)

        fl = open(modelLoadPath, 'rb')
        saveDict = pickle.load(fl)
        fl.close()

        logger.write('Loading Params', 1, self.load.__name__)

        self.memorySize = saveDict['memorySize']
        self.windowSize = saveDict['windowSize']
        self.inputDimension = saveDict['inputDimension']
        self.encoderStateSize = saveDict['encoderStateSize']
        self.lstmStateSize = saveDict['lstmStateSize']
        self.memory = saveDict['memory']
        self.q = saveDict['q']

        self.gruEncoder = tf.keras.layers.GRUCell(units=self.encoderStateSize)
        self.gruEncoder.build(input_shape=(self.inputDimension,))
        self.gruEncoder.set_weights(saveDict['gruEncoder'])

        self.lstm = tf.keras.layers.LSTMCell(units=self.encoderStateSize)
        self.lstm.build(input_shape=(self.inputDimension,))
        self.lstm.set_weights(saveDict['lstm'])

        self.W = tf.Variable(saveDict['W'])
        self.A = tf.Variable(saveDict['A'])
        self.b = tf.Variable(saveDict['b'])

    def prepareData(self, targetSeries, exogenousSeries, logger):
        logger.write('Begin preparing data', 1, self.prepareData.__name__)

        logger.write(f'Target Series Shape: {targetSeries.shape}', 2, self.prepareData.__name__)
        assert (len(targetSeries.shape) == 1)

        trainLength = targetSeries.shape[0] - self.forecastHorizon
        logger.write(f'Train Length: {trainLength}', 2, self.prepareData.__name__)
        assert (trainLength > 0)

        logger.write(f'Exogenous Series: {exogenousSeries}', 2, self.prepareData.__name__)
        if self.inputDimension > 1:
            assert (exogenousSeries is not None)

            logger.write(
                f'Exogenous Series Shape: {exogenousSeries.shape}', 2, self.prepareData.__name__
            )
            assert (self.forecastHorizon + exogenousSeries.shape[0] == targetSeries.shape[0])
            assert (exogenousSeries.shape[1] == self.inputDimension - 1)

            X = np.concatenate(
                [exogenousSeries, np.expand_dims(targetSeries[:trainLength], axis=1)],
                axis=1
            )
        else:
            assert (exogenousSeries is None)
            X = np.expand_dims(targetSeries[:trainLength], axis=1)

        Y = targetSeries[self.forecastHorizon:]

        logger.write(f'X shape: {X.shape}, Y shape: {Y.shape}', 2, self.prepareData.__name__)
        assert (X.shape[0] == Y.shape[0])

        return X, Y

    def trainSequence(self, X, Y, seqStartTime, seqEndTime, optimizer, logger):
        logger.write('Begin Training on Sequence', 1, self.trainSequence.__name__)
        logger.write(
            f'Sequence start: {seqStartTime}, Sequence end: {seqEndTime}',
            2,
            self.trainSequence.__name__
        )

        with tf.GradientTape() as tape:
            self.buildMemory(X, Y, seqStartTime, logger)
            lstmStateList = self.getLstmStates()

            Ypred = []
            for t in range(seqStartTime, seqEndTime + 1):
                pred, lstmStateList = self.predictTimestep(lstmStateList, X, t)
                Ypred.append(pred)

            Ypred = tf.convert_to_tensor(Ypred, dtype=tf.float32)

        loss = tf.keras.losses.MSE(
            Ypred,
            Y[seqStartTime: seqEndTime + 1]
        )

        trainableVars = self.gruEncoder.trainable_variables \
            + self.lstm.trainable_variables \
            + [self.W, self.A, self.b]

        grads = tape.gradient(loss, trainableVars)
        optimizer.apply_gradients(zip(
            grads,
            trainableVars
        ))

        return loss

    def buildMemory(self, X, Y, currentTime, logger):
        if currentTime < self.windowSize:
            raise Exception('Cannot Construct Memory')

        sampleLow = 0
        sampleHigh = currentTime - self.windowSize

        self.memory = [None] * self.memorySize
        self.q = [None] * self.memorySize

        for i in range(self.memorySize):
            windowStartTime = np.random.randint(
                sampleLow,
                sampleHigh + 1
            )

            self.memory[i] = self.runGruOnWindow(X, windowStartTime)
            self.q[i] = Y[windowStartTime + self.windowSize - 1]

        self.memory = tf.stack(self.memory)
        self.q = tf.convert_to_tensor(self.q, dtype=tf.float32)

    def runGruOnWindow(self, X, windowStartTime):
        gruState = self.getGruEncoderState()

        for t in range(
                windowStartTime,
                windowStartTime + self.windowSize
        ):
            gruState, _ = self.gruEncoder(
                np.expand_dims(X[t], 0),
                gruState
            )

        return tf.squeeze(gruState)

    def predictSequence(self, X):
        n = X.shape[0]
        stateList = self.getLstmStates()
        yPred = [None] * n

        for i in range(n):
            yPred[i], stateList = \
                self.predictTimestep(stateList, X, i)

        yPred = np.array(yPred)
        return yPred

    def predictTimestep(self, lstmStateList, X, currentTime):
        [lstmHiddenState, lstmCellState] = self.lstm(
            X[currentTime],
            lstmStateList
        )

        embedding = tf.matmul(
            self.A,
            tf.expand_dims(tf.squeeze(lstmHiddenState), axis=1)
        )
        attentionWeights = self.computeAttention(embedding)

        o1 = tf.squeeze(tf.matmul(
            self.W,
            tf.expand_dims(tf.squeeze(lstmHiddenState), axis=1)
        ))

        o2 = tf.reduce_sum(attentionWeights * self.q)

        bSigmoid = tf.nn.sigmoid(self.b)
        return bSigmoid * o1 + (1 - bSigmoid) * o2, [lstmHiddenState, lstmCellState]

    def computeAttention(self, embedding):
        return tf.squeeze(tf.nn.softmax(tf.linalg.matmul(
            self.memory,
            tf.expand_dims(embedding, axis=1)
        ), axis=0))

    def getLstmStates(self):

        return self.lstm.get_initial_state(
            batch_size=1,
            dtype=tf.float32
        )

    def getGruEncoderState(self):

        return self.gruEncoder.get_initial_state(
            batch_size=1,
            dtype=tf.float32
        )
