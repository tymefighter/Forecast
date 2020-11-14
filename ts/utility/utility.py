import numpy as np


class Utility:

    @staticmethod
    def breakSeq(data, seqLength):
        """
        Break the given numpy array into a list of numpy each of which have number
        of data points equal to provided sequence length except maybe the last, which
        has length less than or equal to the provided length

        :param data: The data, it is a numpy array of shape (n, d) where n is the number
        of data points and d is the number of dimensions
        :param seqLength: Size of each broken up sequence except maybe the last one,
        which would have length less than or equal to seqLength
        :return: Python List of broken up data sequences as numpy arrays of size
        (seqLength, d) each except the last one. The last one has size (lastSize, d)
        where lastSize is always less than or equal to seqLength
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

        if numExoVariables == 0:
            return exogenousSeries is None
        else:
            return \
                exogenousSeries is not None \
                and \
                exogenousSeries.shape[1] == numExoVariables
