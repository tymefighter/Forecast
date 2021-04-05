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
        https://en.wikipedia.org/wiki/Root-mean-square_deviation

        :param forecastSeq: Forecast (Predicted) sequence, a numpy array
        of shape (n, d)
        :param actualSeq: Actual sequence, a numpy array of shape (n, d)
        :return: Returns RMSE for each of the 'd' components as a numpy
        array of shape (d,)
        """

        assert forecastSeq.shape == actualSeq.shape

        return np.sqrt(Metric.mse(forecastSeq, actualSeq))

    @staticmethod
    def errorOnExtremes(
            forecastSeq: np.ndarray,
            actualSeq: np.ndarray,
            extremeThreshold, func
    ):
        """
        Compute any provided error on only extreme values i.e.
        timesteps corresponding to values in the actual sequence which
        exceed the given threshold.

        :param forecastSeq: Forecast (Predicted) sequence, a numpy array
        of shape (n, d)
        :param actualSeq: Actual sequence, a numpy array of shape (n, d)
        :param extremeThreshold: Timesteps whose values exceed this value
        are termed extreme, and other timesteps are termed normal
        :param func: Function which is to be applied on the extreme
        subsequence of the provided time series
        :return: The output of the provided function on the extreme
        subsequence
        """

        maskArr = actualSeq > extremeThreshold

        return func(forecastSeq[maskArr], actualSeq[maskArr])
