import numpy as np
from ts.decompose import MovingAverage


def isOdd(n: int) -> bool: return (n & 1) != 0


class ClassicalDecompose:

    @staticmethod
    def decompose(timeSeries, seasonalPeriod):
        """
        Decompose the provided time series using the classical
        time series decomposition method.
        Link: https://otexts.com/fpp2/classical-decomposition.html

        :param timeSeries: time series which is to be decomposed
        :param seasonalPeriod: the seasonality period
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
        seasonalitySeries = timeSeries - trendSeries

        for currSeason in range(seasonalPeriod):
            numCurrSeasonValues = (n - currSeason - 1) // seasonalPeriod
            currSeasonValues = np.zeros((numCurrSeasonValues, d))

            for i in range(numCurrSeasonValues):
                currSeasonValues[i] = \
                    seasonalitySeries[currSeason + i * seasonalPeriod]

            seasonalValues[currSeason, :] = currSeasonValues.mean(axis=0)

        for i in range(n):
            seasonalitySeries[i] = seasonalValues[i % seasonalPeriod]

        remainderSeries = timeSeries - trendSeries - seasonalitySeries

        return trendSeries, seasonalitySeries, remainderSeries
