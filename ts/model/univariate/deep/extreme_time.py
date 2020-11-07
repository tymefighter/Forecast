import pickle
import tensorflow as tf

from ts.model.univariate.univariate import UnivariateModel
from ts.log import DEFAULT_LOG_PATH, FileLogger


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
            logger.write('Initializing Members', 2, self.__init__.__name__)

            self.forecastHorizon = forecastHorizon
            self.memorySize = memorySize
            self.windowSize = windowSize
            self.encoderStateSize = encoderStateSize
            self.lstmStateSize = lstmStateSize
            self.inputDimension = numExoVariables + 1
            self.memory = None
            self.q = None

            logger.write('Building Model Parameters', 2, self.__init__.__name__)

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
            modelSavePath=None,
            verbose=1,
            logPath=DEFAULT_LOG_PATH,
            logLevel=1
    ):
        """
        Train the Model Parameters on the provided data

        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (n,)
        :param sequenceLength: Length of each training sequence
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (n, self.numExoVariables), it can be None only if
        self.numExoVariables is 0 in which case the exogenous variables are not
        considered
        :param modelSavePath: Path where to save the model parameters after
        each training an a sequence, if None then parameters are not saved
        :param verbose: Verbose level, 0 is nothing, greater values increases
        the information printed to the console
        :param logPath: Path where to log the information
        :param logLevel: Logging level, 0 means no logging, greater values indicate
        more information
        :return: None
        """
        pass

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

        if self.memory is None:
            logger.write('Memory not constructed - cannot save model', 1, self.save.__name__)
            raise Exception('Memory not constructed - cannot save model')

        logger.write('Constructing Dictionary from model params', 2, self.save.__name__)

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

        logger.write('Saving Dictionary', 2, self.save.__name__)

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
        logger.write('Load Dictionary from Model Params file', 2, self.load.__name__)

        fl = open(modelLoadPath, 'rb')
        saveDict = pickle.load(fl)
        fl.close()

        logger.write('Loading Params', 2, self.load.__name__)

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

    def trainSequence(self, X, Y, seqStartTime, seqEndTime, logger):
        pass

    def buildMemory(self, X, Y, currentTime):
        pass

    def runGruOnWindow(self, X, windowStartTime):
        pass

    def predictSequence(self, X):
        pass

    def predictTimestep(self, lstmStateList, X, currentTime):
        pass

    def computeAttention(self, embedding):
        pass


