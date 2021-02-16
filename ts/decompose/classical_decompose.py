import numpy as np
from ts.decompose import MovingAverage


def isOdd(n: int) -> bool: return (n & 1) != 0


class ClassicalDecompose:

    @staticmethod
    def decompose(
            timeSeries: np.ndarray,
            seasonalPeriod: int, additive: bool = True
    ) -> (np.ndarray, np.ndarray, np.ndarray):
        """
        Decompose the provided time series using the classical
        time series decomposition method.
        Link: https://otexts.com/fpp2/classical-decomposition.html

        :param timeSeries: time series which is to be decomposed
        :param seasonalPeriod: the seasonality period
        :param additive: if True, then the model is assumed to be
        additive, else it is assumed to be multiplicative
        :return: (trend series, seasonality series, remainder series)
        """

        if isOdd(seasonalPeriod):
            trendSeries = MovingAverage.movingAverage(timeSeries, seasonalPeriod)
            numEleRemove = (seasonalPeriod - 1) // 2
        else:
            trendSeries = MovingAverage\
                .doubleMovingAverage(timeSeries, seasonalPeriod, 2)
            numEleRemove = seasonalPeriod // 2

        timeSeries = timeSeries[numEleRemove: -numEleRemove]

        (n, d) = timeSeries.shape
        seasonalValues = np.zeros((seasonalPeriod, d))

        if additive:
            seasonalitySeries = timeSeries - trendSeries
        else:
            seasonalitySeries = timeSeries / trendSeries

        for currSeason in range(seasonalPeriod):
            numCurrSeasonValues = (n - currSeason - 1) // seasonalPeriod
            currSeasonValues = np.zeros((numCurrSeasonValues, d))

            for i in range(numCurrSeasonValues):
                currSeasonValues[i] = \
                    seasonalitySeries[currSeason + i * seasonalPeriod]

            seasonalValues[currSeason, :] = currSeasonValues.mean(axis=0)

        seasonalValues -= seasonalValues.mean(axis=0)
        if not additive:
            seasonalValues += 1

        for i in range(n):
            seasonalitySeries[i] = seasonalValues[i % seasonalPeriod]

        if additive:
            remainderSeries = timeSeries - trendSeries - seasonalitySeries
        else:
            remainderSeries = timeSeries / (trendSeries * seasonalitySeries)

        return trendSeries, seasonalitySeries, remainderSeries
