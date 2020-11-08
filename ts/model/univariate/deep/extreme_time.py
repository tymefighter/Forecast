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
            logger.log('Initializing Members', 1, self.__init__.__name__)

            self.forecastHorizon = forecastHorizon
            self.memorySize = memorySize
            self.windowSize = windowSize
            self.encoderStateSize = encoderStateSize
            self.lstmStateSize = lstmStateSize
            self.inputDimension = numExoVariables + 1
            self.memory = None
            self.q = None

            logger.log('Building Model Parameters', 1, self.__init__.__name__)

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
            logLevel=1,
            returnLosses=True
    ):
        """
        Train the Model Parameters on the provided data

        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (n + self.forecastHorizon,)
        :param sequenceLength: Length of each training sequence
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (n, numExoVariables), it can be None only if
        numExoVariables is 0 in which case the exogenous variables are not
        considered
        :param optimizer: Optimizer of training the parameters
        :param modelSavePath: Path where to save the model parameters after
        each training an a sequence, if None then parameters are not saved
        :param verboseLevel: Verbose level, 0 is nothing, greater values increases
        the information printed to the console
        :param logPath: Path where to log the information
        :param logLevel: Logging level, 0 means no logging, greater values indicate
        more information
        :param returnLosses: If True, then losses are returned, else losses are not
        returned
        :return: If returnLosses is True, then numpy array of losses of shape (numSeq,)
        is returned, else None is returned
        """

        logger = FileLogger(logPath, logLevel)
        verbose = ConsoleLogger(verboseLevel)

        X, Y = self.prepareData(targetSeries, exogenousSeries, logger)

        seqStartTime = self.windowSize
        n = X.shape[0]
        logger.log(f'Seq Start Time: {seqStartTime}, Train len: {n}', 2, self.train.__name__)
        assert (seqStartTime < n)

        logger.log('Begin Training', 1, self.train.__name__)

        if returnLosses:
            losses = []

        while seqStartTime < n:
            seqEndTime = min(seqStartTime + sequenceLength, n - 1)

            startTime = time.time()
            loss = self.trainSequence(X, Y, seqStartTime, seqEndTime, optimizer, logger)
            endTime = time.time()
            timeTaken = endTime - startTime

            verbose.log(f'start timestep: {seqStartTime}'
                        + f' | end timestep: {seqEndTime}'
                        + f' | time taken: {timeTaken : .2f} sec'
                        + f' | Loss: {loss}', 1)

            if returnLosses:
                losses.append(loss)

            if modelSavePath is not None:
                logger.log(f'Saving Model at {modelSavePath}', 1, self.train.__name__)

        self.buildMemory(X, Y, n, logger)
        logger.close()

        if returnLosses:
            return np.array(losses)

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
        should be a numpy array of shape (n + self.forecastHorizon,)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (n, numExoVariables), it can be None only if
        numExoVariables is 0 in which case the exogenous variables are not
        considered
        :param logPath: Path where to log the information
        :param logLevel: Logging level, 0 means no logging, greater values indicate
        more information
        :return: Forecast targets predicted by the model, it has shape (n,), the
        horizon of the targets is the same as self.forecastHorizon
        """

        logger = FileLogger(logPath, logLevel)
        logger.log('Begin Prediction', 1, self.trainSequence.__name__)

        X = self.preparePredictData(targetSeries, exogenousSeries, logger)

        n = X.shape[0]
        lstmStateList = self.getInitialLstmStates()
        Ypred = [None] * n

        logger.log(f'LSTM state shapes: {lstmStateList[0].shape}, {lstmStateList[1].shape}', 2,
                   self.trainSequence.__name__)

        for i in range(n):
            Ypred[i], lstmStateList = \
                self.predictTimestep(lstmStateList, X, i)

        Ypred = np.array(Ypred)
        logger.log(f'Output Shape: {Ypred.shape}', 2, self.trainSequence.__name__)

        return Ypred

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
        numpy array of shape (numTimesteps, numExoVariables), it can be None
        only if numExoVariables is 0 in which case the exogenous variables
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
        logger.log(f'Memory Shape: {self.memory.shape}', 2, self.save.__name__)

        logger.log('Constructing Dictionary from model params', 1, self.save.__name__)

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

        logger.log('Saving Dictionary', 1, self.save.__name__)

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
        logger.log('Load Dictionary from Model Params file', 1, self.load.__name__)

        fl = open(modelLoadPath, 'rb')
        saveDict = pickle.load(fl)
        fl.close()

        logger.log('Loading Params', 1, self.load.__name__)

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

    def preparePredictData(self, targetSeries, exogenousSeries, logger):
        """
        Prepare the data for training

        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (n,)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (n, numExoVariables), it can be None only if
        numExoVariables is 0 in which case the exogenous variables are not
        considered
        :param logger: The logger which would be used to log information
        :return: prepared feature data X, X has shape (n, numExoVariables + 1), X can
        also be said to have shape (n, self.inputShape) since
        self.inputShape = numExoVariables + 1s
        """

        logger.log('Begin preparing data', 1, self.preparePredictData.__name__)
        logger.log(f'Target Series Shape: {targetSeries.shape}', 2, self.preparePredictData.__name__)
        assert (len(targetSeries.shape) == 1)

        trainLength = targetSeries.shape[0] - self.forecastHorizon

        logger.log(f'Train Length: {trainLength}', 2, self.preparePredictData.__name__)
        assert (trainLength > 0)

        logger.log(f'Exogenous Series: {exogenousSeries}', 2, self.preparePredictData.__name__)
        if self.inputDimension > 1:
            assert (exogenousSeries is not None)

            logger.log(f'Exogenous Series Shape: {exogenousSeries.shape}', 2, self.preparePredictData.__name__)
            assert (exogenousSeries.shape[0] == targetSeries.shape[0])
            assert (exogenousSeries.shape[1] == self.inputDimension - 1)

            X = np.concatenate(
                [exogenousSeries, np.expand_dims(targetSeries[:trainLength], axis=1)],
                axis=1
            )
        else:
            assert (exogenousSeries is None)
            X = np.expand_dims(targetSeries[:trainLength], axis=1)

        return X

    def prepareData(self, targetSeries, exogenousSeries, logger):
        """
        Prepare the data for training

        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (n + self.forecastHorizon,)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (n, numExoVariables), it can be None only if
        numExoVariables is 0 in which case the exogenous variables are not
        considered
        :param logger: The logger which would be used to log information
        :return: prepared data X, Y as features and targets, X has shape
        (n, numExoVariables + 1), Y has shape (n,). X can also be said to have
        shape (n, self.inputShape) since self.inputShape = numExoVariables + 1s
        """

        logger.log('Begin preparing data', 1, self.prepareData.__name__)
        logger.log(f'Target Series Shape: {targetSeries.shape}', 2, self.prepareData.__name__)
        assert (len(targetSeries.shape) == 1)

        trainLength = targetSeries.shape[0] - self.forecastHorizon

        logger.log(f'Train Length: {trainLength}', 2, self.prepareData.__name__)
        assert (trainLength > 0)

        logger.log(f'Exogenous Series: {exogenousSeries}', 2, self.prepareData.__name__)
        if self.inputDimension > 1:
            assert (exogenousSeries is not None)

            logger.log(f'Exogenous Series Shape: {exogenousSeries.shape}', 2, self.prepareData.__name__)
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

        logger.log(f'X shape: {X.shape}, Y shape: {Y.shape}', 2, self.prepareData.__name__)
        assert (X.shape[0] == Y.shape[0])

        return X, Y

    def trainSequence(self, X, Y, seqStartTime, seqEndTime, optimizer, logger):
        """

        :param X: Features, has shape (n, self.inputShape)
        :param Y: Targets, has shape (n,)
        :param seqStartTime: Sequence Start Time
        :param seqEndTime: Sequence End Time
        :param optimizer: The optimization algorithm
        :param logger: The logger which would be used to log information
        :return: The loss value resulted from training on the sequence
        """

        logger.log('Begin Training on Sequence', 1, self.trainSequence.__name__)
        logger.log(f'Sequence start: {seqStartTime}, Sequence end: {seqEndTime}', 2, self.trainSequence.__name__)

        with tf.GradientTape() as tape:
            self.buildMemory(X, Y, seqStartTime, logger)
            lstmStateList = self.getInitialLstmStates()

            logger.log(f'LSTM state shapes: {lstmStateList[0].shape}, {lstmStateList[1].shape}', 2,
                       self.trainSequence.__name__)

            Ypred = []
            for t in range(seqStartTime, seqEndTime + 1):
                pred, lstmStateList = self.predictTimestep(lstmStateList, X, t, logger)
                Ypred.append(pred)

            Ypred = tf.convert_to_tensor(Ypred, dtype=tf.float32)
            logger.log(f'Prediction Shape: {Ypred.shape}', 2, self.trainSequence.__name__)

        loss = tf.keras.losses.MSE(
            Ypred,
            Y[seqStartTime: seqEndTime + 1]
        )
        logger.log(f'Loss: {loss}', 2, self.trainSequence.__name__)

        trainableVars = self.gruEncoder.trainable_variables \
            + self.lstm.trainable_variables \
            + [self.W, self.A, self.b]

        logger.log('Performing Gradient Descent', 1, self.trainSequence.__name__)

        grads = tape.gradient(loss, trainableVars)
        assert(len(trainableVars) == len(grads))

        optimizer.apply_gradients(zip(
            grads,
            trainableVars
        ))

        return loss

    def buildMemory(self, X, Y, currentTime, logger):
        """
        Build Model Memory using the timesteps seen up till now

        :param X: Features, has shape (n, self.inputShape)
        :param Y: Targets, has shape (n,)
        :param currentTime: current timestep, memory would be built only using the
        timestep earlier than the current timestep
        :param logger: The logger which would be used to log information
        :return: None
        """

        logger.log(f'Building Memory', 1, self.buildMemory.__name__)
        logger.log(f'Current Time: {currentTime}', 2, self.buildMemory.__name__)
        assert(currentTime >= self.windowSize)

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

        logger.log(f'Memory Shape: {self.memory.shape}, Out Shape: {self.q.shape}', 2, self.buildMemory.__name__)

    def runGruOnWindow(self, X, windowStartTime, logger):
        """
        Runs GRU on the window and returns the final state

        :param X: Features, has shape (n, self.inputShape)
        :param windowStartTime: Starting timestep of the window
        :param logger: The logger which would be used to log information
        :return: The final state after running on the window, it has shape (self.encoderStateSize,)
        """

        logger.log(f'Window Start Time: {windowStartTime}', 2, self.runGruOnWindow.__name__)

        gruState = self.getInitialGruEncoderState()

        for t in range(
                windowStartTime,
                windowStartTime + self.windowSize
        ):
            gruState, _ = self.gruEncoder(
                np.expand_dims(X[t], 0),
                gruState
            )

        finalState = tf.squeeze(gruState)
        logger.log(f'GRU final state shape: {finalState.shape}', 2, self.runGruOnWindow.__name__)

        return finalState

    def predictTimestep(self, lstmStateList, X, currentTime, logger):
        """
        Predict on a Single Timestep

        :param lstmStateList: List of the current two states the LSTM requires
        :param X: Features, has shape (n, self.inputShape)
        :param currentTime: Current Timestep
        :param logger: The logger which would be used to log information
        :return: The predicted value on current timestep
        """

        logger.log(f'LSTM state shapes: {lstmStateList[0].shape}, {lstmStateList[1].shape}', 2,
                   self.predictTimestep.__name__)

        [lstmHiddenState, lstmCellState] = self.lstm(
            X[currentTime],
            lstmStateList
        )

        embedding = tf.matmul(
            self.A,
            tf.expand_dims(tf.squeeze(lstmHiddenState), axis=1)
        )
        logger.log(f'Embedding Shape: {embedding.shape}', 2, self.predictTimestep.__name__)

        attentionWeights = self.computeAttention(embedding)
        logger.log(f'Attention Shape: {attentionWeights.shape}', 2, self.predictTimestep.__name__)

        o1 = tf.squeeze(tf.matmul(
            self.W,
            tf.expand_dims(tf.squeeze(lstmHiddenState), axis=1)
        ))
        logger.log(f'Output1: {o1}', 2, self.predictTimestep.__name__)

        o2 = tf.reduce_sum(attentionWeights * self.q)
        logger.log(f'Output2: {o1}', 2, self.predictTimestep.__name__)

        bSigmoid = tf.nn.sigmoid(self.b)
        pred = bSigmoid * o1 + (1 - bSigmoid) * o2, [lstmHiddenState, lstmCellState]

        logger.log(f'Prediction: {pred}', 2, self.predictTimestep.__name__)

        return pred

    def computeAttention(self, embedding):
        """
        Computes Attention Weights by taking softmax of the inner product
        between embedding of the input and the memory states

        :param embedding: Embedding of the input
        :return: Attention Weight Values
        """

        return tf.nn.softmax(tf.squeeze(tf.linalg.matmul(
            self.memory,
            tf.expand_dims(embedding, axis=1)
        )))

    def getInitialLstmStates(self):
        """
        Computes Initial LSTM States (i.e. both of the initial states)

        :return: Initial LSTM State List
        """

        return self.lstm.get_initial_state(
            batch_size=1,
            dtype=tf.float32
        )

    def getInitialGruEncoderState(self):
        """
        Computes Initial GRU Encoder State

        :return: Initial GRU State
        """

        return self.gruEncoder.get_initial_state(
            batch_size=1,
            dtype=tf.float32
        )
