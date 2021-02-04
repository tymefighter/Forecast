import numpy as np


class Utility:

    @staticmethod
    def breakSeq(data, seqLength):
        """
        Break the given numpy array into a list of numpy each of which have number
        of data points equal to provided sequence length except maybe the last, which
        has length less than or equal to the provided length

        :param data: The data, it is a numpy array of shape (n, d) where n is the number
        of data points and d is the number of dimensions or a numpy array of shape (n,)
        :param seqLength: Size of each broken up sequence except maybe the last one,
        which would have length less than or equal to seqLength
        :return: Python List of broken up data sequences as numpy arrays of size
        (seqLength, d) each except the last one, the last one has size (lastSize, d)
        where lastSize is always less than or equal to seqLength, this is when the
        input has shape (n, d). If input has shape (n,), then the dimension axis is
        not present, i.e. each seq has shape (seqLength,), last one has (lastSize,)
        """

        n = data.shape[0]
        dataSeq = []
        seqStart = 0

        while seqStart < n:
            seqEnd = min(seqStart + seqLength, n)
            dataSeq.append(data[seqStart:seqEnd])
            seqStart = seqEnd

        return dataSeq

    @staticmethod
    def breakTrainSeq(targetSeries, exogenousSeries, seqLength, forecastHorizon=None):
        """
        Break Target Series and Exogenous Series into a training list of
        sequences which is required by models which train on multiple sequences

        :param targetSeries: The target Series, it has shape (n, d1) or (n,)
        :param exogenousSeries: Exogenous Series, it can be None, and if it is not
        None, then it has shape (n, d2)
        :param seqLength: The length of each broken sequence in the returned list
        of sequences
        :param forecastHorizon: The forecast horizon, it's not used when
        exogenousSeries is None (hence can be None), else it cannot be Nones
        :return: If exogenous series is None, then breaks the target series and
        returns a list of parts of the target series only. If exogenous series
        is not None, then returns a list of tuples where the first element of the
        tuple is a target series part, and second is a exogenous series part, and
        the target series part has 'forecastHorizon' additional number of elements
        than the exogenous series part of each tuple
        """

        if exogenousSeries is None:
            return Utility.breakSeq(targetSeries, seqLength)
        else:
            assert (
                targetSeries.shape[0] == exogenousSeries.shape[0]
                and forecastHorizon is not None
            )

        n = targetSeries.shape[0]
        targetSequences = []
        exoSequences = []
        seqStart = 0

        while seqStart < n:
            targetSeqEnd = min(seqStart + seqLength + forecastHorizon, n)
            exoSeqEnd = targetSeqEnd - forecastHorizon
            if exoSeqEnd <= seqStart:
                break

            targetSequences.append(targetSeries[seqStart:targetSeqEnd])
            exoSequences.append(exogenousSeries[seqStart:exoSeqEnd])
            seqStart = exoSeqEnd

        assert (len(targetSequences) == len(exoSequences))

        return list(zip(targetSequences, exoSequences))

    @staticmethod
    def trainTestSplit(data, train, val=None):
        """
        Split the data into training and testing data, or training, testing and validation
        data. Important - data is not shuffled since it is assumed to be a time series data

        :param data: The data, it is a numpy array of shape (n, d) where n is the number
        of data points and d is the number of dimensions
        :param train: If is is a float between 0 and 1, then it is fraction of training
        data, If >= 1, then it is the number of training samples
        :param val: If None, then split is only train and test, If it is a float
        between 0 and 1, then it is fraction of validation data, If >= 1, then it is
        the number of validation samples
        :return: If val is None, two return value - train and test data. If val is not
        None, then three return values - train, val, test
        """

        n = data.shape[0]

        if 0 <= train < 1:
            nTrain = round(train * n)
        else:
            nTrain = train

        assert (nTrain <= n)

        if val is None:
            dataTrain = data[:nTrain]
            dataTest = data[nTrain:]

            return dataTrain, dataTest

        if 0 <= val < 1:
            nVal = round(val * n)
        else:
            nVal = val

        assert (nTrain + nVal <= n)

        dataTrain = data[:nTrain]
        dataVal = data[nTrain:nTrain + nVal]
        dataTest = data[nTrain + nVal:]

        return dataTrain, dataVal, dataTest

    @staticmethod
    def trainTestSplitSeries(targetSeries, exogenousSeries, train, val=None):
        """
        Split the data into training and testing data, or training, testing and validation
        data. Important - data is not shuffled since it is assumed to be a time series data

        :param targetSeries: Target series, it is a numpy array of shape (n, d1)
        where n is the number of data points and d1 is the number of dimensions
        :param exogenousSeries: Exogenous series, it is a numpy array of shape (n, d2) where
        n is the number of data points and d2 is the number of dimensions. Cannot be None,
        if you don't have an exogenous series, then please use the trainTestSplit function.
        :param train: If is is a float between 0 and 1, then it is fraction of training
        data, If >= 1, then it is the number of training samples
        :param val: If None, then split is only train and test, If it is a float
        between 0 and 1, then it is fraction of validation data, If >= 1, then it is
        the number of validation samples
        :return: If val is None, then returns (targetTrain, exoTrain), (exoTrain, exoTest),
        else returns (targetTrain, exoTrain), (targetVal, exoVal), (targetTest, exoTest)
        where (targetTrain, exoTrain) is the training set, (targetVal, exoVal) is the
        validation set and (targetTest, exoTest) is the test set
        """

        if val is None:
            targetTrain, targetTest = Utility.trainTestSplit(targetSeries, train)
            exoTrain, exoTest = Utility.trainTestSplit(exogenousSeries, train)
            return (targetTrain, exoTrain), (targetTest, exoTest)

        targetTrain, targetVal, targetTest = \
            Utility.trainTestSplit(targetSeries, train, True)

        exoTrain, exoVal, exoTest = \
            Utility.trainTestSplit(exogenousSeries, train, True)

        return (targetTrain, exoTrain), (targetVal, exoVal), (targetTest, exoTest)

    @staticmethod
    def prepareDataPred(targetSeries, exogenousSeries):
        """
        Prepare Forecasting Data During Prediction time

        :param targetSeries: Target Series, it has shape (n, d1) or (n,)
        :param exogenousSeries: Exogenous Series, it is None or it has shape (n, d2)
        :return: Feature data of shape (n, d1 + d2), when targetSeries has shape (n,),
        we have d1 = 1, when exogenousSeries is None, we have d2 = 0
        """

        assert (
                exogenousSeries is None
                or
                (targetSeries.shape[0] == exogenousSeries.shape[0])
        )

        if len(targetSeries.shape) == 1:
            targetSeries = np.expand_dims(targetSeries, axis=1)

        if exogenousSeries is None:
            X = targetSeries
        else:
            X = np.concatenate([targetSeries, exogenousSeries], axis=1)

        return X

    @staticmethod
    def prepareDataTrain(targetSeries, exogenousSeries, forecastHorizon):
        """

        :param targetSeries: Target Series, it has shape (n + forecastHorizon, d1)
        or (n + forecastHorizon,)
        :param exogenousSeries: Exogenous Series, it is None or it has shape (n, d2)
        :param forecastHorizon: How much further in the future the model has to
        predict the target series variable
        :return: Feature data of shape (n, d1 + d2) when targetSeries has shape
        (n + forecastHorizon,) we have d1 = 1, when exogenousSeries is None,
        we have d2 = 0 and Target data of shape (n, d1) if targetSeries has
        shape (n + forecastHorizon, d1) else (n,)
        """

        assert (targetSeries.shape[0] > forecastHorizon)

        assert (
                exogenousSeries is None
                or
                (targetSeries.shape[0] == exogenousSeries.shape[0] + forecastHorizon)
        )

        n = targetSeries.shape[0] - forecastHorizon

        X = Utility.prepareDataPred(targetSeries[:n], exogenousSeries)
        Y = targetSeries[forecastHorizon:]

        assert (X.shape[0] == Y.shape[0])
        assert (len(Y.shape) == len(targetSeries.shape))

        return X, Y

    @staticmethod
    def isExoShapeValid(exogenousSeries, numExoVariables):
        """
        Checks if exogenous series shape is valid

        :param exogenousSeries: The exogenous variables time series, it can be
        None or it is a numpy array of shape (n, d)
        :param numExoVariables: Number of exogenous variables
        :return: If numExogenousVariables is 0, then the exogenous series should
        be None, else d must be numExoVariables. If this is satified then returns
        True, else returns False
        """

        if numExoVariables == 0:
            return exogenousSeries is None
        else:
            return \
                exogenousSeries is not None \
                and \
                exogenousSeries.shape[1] == numExoVariables

    @staticmethod
    def generateMultipleSequence(
            dataGenerator,
            numSequences,
            minSequenceLength,
            maxSequenceLength
    ):
        """
        Generates multiple sequences using the provided data generator

        :param dataGenerator: The data generator
        :param numSequences: Number of sequences to generate
        :param minSequenceLength: Minimum sequence length - lower bound
        on the sequence length
        :param maxSequenceLength: Maximum sequence length - upper bound
        on the sequence length
        :return: List of generated sequences satisfying the conditions
        provided in the input
        """

        assert (minSequenceLength <= maxSequenceLength)

        return [
            dataGenerator.generate(seqLength)
            for seqLength in list(np.random.randint(
                low=minSequenceLength,
                high=maxSequenceLength + 1,
                size=(numSequences,)
            ))
        ]
