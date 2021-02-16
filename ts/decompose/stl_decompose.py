import numpy as np
from statsmodels.tsa.seasonal import STL


class StlDecompose:

    @staticmethod
    def decompose(
            timeSeries: np.ndarray,
            period: int, seasonal: int,
            robust: bool
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Decompose the provided time series using the STL
        time series decomposition method.
        Links:
        - https://otexts.com/fpp2/stl.html
        - https://www.statsmodels.org/stable/examples/notebooks/generated/stl_decomposition.html

        :param timeSeries: time series which is to be decomposed, it is
        a numpy array of shape (n, d)
        :param period: periodicity of the sequence
        :param seasonal: length of seasonal smoother
        :param robust: if True, then uses weighted version which is robust
        to outliers and extreme values
        :return: (trend series, seasonality series, remainder series),
        each element of this 3-tuple is a numpy array of shape (n, d)
        """

        d = timeSeries.shape[1]

        trendArr = []
        seasonalArr = []
        remainArr = []

        for i in range(d):
            decomposeResult = STL(
                endog=timeSeries[:, i],
                period=period,
                seasonal=seasonal,
                robust=robust
            ).fit()

            trendArr.append(np.expand_dims(decomposeResult.trend, axis=1))
            seasonalArr.append(np.expand_dims(decomposeResult.seasonal, axis=1))
            remainArr.append(np.expand_dims(decomposeResult.resid, axis=1))

        trendSeries = np.concatenate(trendArr, axis=1)
        seasonalitySeries = np.concatenate(seasonalArr, axis=1)
        remainderSeries = np.concatenate(remainArr, axis=1)

        return trendSeries, seasonalitySeries, remainderSeries
