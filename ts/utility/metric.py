import numpy as np


class Metric:

    @staticmethod
    def mape(forecastSeq: np.ndarray, actualSeq: np.ndarray):
        """
        Mean Absolute Percentage Error
        https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecastSeq: Forecast (Predicted) sequence, a numpy array
        of shape (n, d)
        :param actualSeq: Actual sequence, a numpy array of shape (n, d)
        :return: Returns MAPE for each of the 'd' components as a numpy
        array of shape (d,)
        """

        assert forecastSeq.shape == actualSeq.shape

        return 100. * np.mean(np.abs((actualSeq - forecastSeq) / actualSeq), axis=0)

    @staticmethod
    def mae(forecastSeq: np.ndarray, actualSeq: np.ndarray):
        """
        Mean Absolute Error
        https://en.wikipedia.org/wiki/Mean_absolute_error

        :param forecastSeq: Forecast (Predicted) sequence, a numpy array
        of shape (n, d)
        :param actualSeq: Actual sequence, a numpy array of shape (n, d)
        :return: Returns MAE for each of the 'd' components as a numpy
        array of shape (d,)
        """

        assert forecastSeq.shape == actualSeq.shape

        return np.mean(np.abs(actualSeq - forecastSeq), axis=0)

    @staticmethod
    def mpe(forecastSeq: np.ndarray, actualSeq: np.ndarray):
        """
        Mean Percentage Error
        https://en.wikipedia.org/wiki/Mean_percentage_error

        :param forecastSeq: Forecast (Predicted) sequence, a numpy array
        of shape (n, d)
        :param actualSeq: Actual sequence, a numpy array of shape (n, d)
        :return: Returns MPE for each of the 'd' components as a numpy
        array of shape (d,)
        """

        assert forecastSeq.shape == actualSeq.shape

        return 100. * np.mean((actualSeq - forecastSeq) / actualSeq, axis=0)

    @staticmethod
    def mse(forecastSeq: np.ndarray, actualSeq: np.ndarray):
        """
        Mean Squared Error
        https://en.wikipedia.org/wiki/Mean_squared_error

        :param forecastSeq: Forecast (Predicted) sequence, a numpy array
        of shape (n, d)
        :param actualSeq: Actual sequence, a numpy array of shape (n, d)
        :return: Returns MSE for each of the 'd' components as a numpy
        array of shape (d,)
        """

        assert forecastSeq.shape == actualSeq.shape

        return np.mean(np.square(actualSeq - forecastSeq), axis=0)

    @staticmethod
    def rmse(forecastSeq: np.ndarray, actualSeq: np.ndarray):
        """
        Root Mean Squared Error

        :param forecastSeq: Forecast (Predicted) sequence, a numpy array
        of shape (n, d)
        :param actualSeq: Actual sequence, a numpy array of shape (n, d)
        :return: Returns RMSE for each of the 'd' components as a numpy
        array of shape (d,)
        """

        assert forecastSeq.shape == actualSeq.shape

        return np.sqrt(Metric.mse(forecastSeq, actualSeq))
