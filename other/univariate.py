from ts.log import DEFAULT_LOG_PATH


class UnivariateModel:

    def train(
            self,
            targetSeries,
            sequenceLength,
            exogenousSeries=None,
            modelSavePath=None,
            verboseLevel=1,
            returnLosses=True
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
        :param verboseLevel: Verbose level, 0 is nothing, greater values increases
        the information printed to the console
        :param returnLosses: If True, then losses are returned, else losses are not
        returned
        :return: If returnLosses is True, then numpy array of losses of shape (numSeq,)
        is returned, else None is returned
        """
        pass

    def predict(
            self,
            targetSeries,
            exogenousSeries=None
    ):
        """
        Forecast using the model parameters on the provided input data

        :param targetSeries: Univariate Series of the Target Variable, it
        should be a numpy array of shape (n,)
        :param exogenousSeries: Series of exogenous Variables, it should be a
        numpy array of shape (n, self.numExoVariables), it can be None only if
        self.numExoVariables is 0 in which case the exogenous variables are not
        considered
        :return: Forecast targets predicted by the model, it has shape (n,), the
        horizon of the targets is the same as self.forecastHorizon
        """
        pass

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
        numpy array of shape (numTimesteps, self.numExoVariables), it can be None
        only if self.numExoVariables is 0 in which case the exogenous variables
        are not considered
        :param returnPred: If True, then return predictions along with loss, else
        return on loss
        :return: If True, then return predictions along with loss of the predicted
        and true targets, else return only loss
        """
        pass

    def save(
            self,
            modelSavePath
    ):
        """
        Save the model parameters at the provided path

        :param modelSavePath: Path where the parameters are to be saved
        :return: None
        """
        pass

    def load(
            self,
            modelLoadPath
    ):
        """
        Load the model parameters from the provided path

        :param modelLoadPath: Path from where the parameters are to be loaded
        :return: None
        """
        pass
