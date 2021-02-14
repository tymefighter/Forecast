import numpy as np


def isOdd(n: int) -> bool: return (n & 1) != 0


class MovingAverage:

    @staticmethod
    def movingAverage(
            timeSeries: np.ndarray,
            order: int,
            zeroPad: bool = False,
            prevMore: bool = True
    ) -> np.ndarray:
        """
        Compute moving average of the time series method.
        Link: https://otexts.com/fpp2/moving-averages.html

        :param timeSeries: time series whose moving average is to be
        estimated. It is a numpy array of shape (n, d)
        :param order: order of moving average which is to be used. If
        order is odd, then for computing the value corresponding to
        timestep t, we compute the mean of timesteps
        {t - (order - 1) / 2, .. t, .., t + (order - 1) / 2}, else
        if the order is even, we use timesteps
        {t - numPrev, .. t, .., t + numNext} where numPrev and
        numNext depend on the value of 'prevMore' argument
        :param zeroPad: If this is True, then the returned time
        series has the same length since zero padding is performed
        before averaging, else the returned time series
        has shape (n - order + 1, d), since we do not compute
        value corresponding the left and right edges.
        :param prevMore: only used if order is even, in this case
        if prevMore is True, then numPrev = order / 2 and
        numNext = order / 2 - 1, else it is just the reverse
        :return: the moving average time series
        """

        if isOdd(order):
            numPrev = numNext = order // 2
        else:
            if prevMore:
                numPrev = order // 2
                numNext = numPrev - 1
            else:
                numNext = order // 2
                numPrev = numNext - 1

        n, d = timeSeries.shape

        # ma: moving average
        maSeriesLength = n if zeroPad else max(n - numPrev - numNext, 0)
        maSeries = np.zeros((maSeriesLength, d))

        idx = 0
        for t in range(n):

            if t < numPrev or n - t <= numNext:
                if not zeroPad:
                    continue
                else:

                    if t < numPrev:
                        leftIdx = 0
                        leftArr = np.zeros((numPrev - t, d))
                    else:
                        leftIdx = t - numPrev
                        leftArr = np.zeros((0, d))

                    if n - t <= numNext:
                        rightIdx = n - 1
                        rightArr = np.zeros((t + numNext - n + 1, d))
                    else:
                        rightIdx = t + numNext
                        rightArr = np.zeros((0, d))

                    maSeries[idx, :] = np.concatenate((
                        leftArr, timeSeries[leftIdx: rightIdx], rightArr
                    ), axis=0).mean(axis=0, keepdims=True)

            else:
                maSeries[idx, :] = timeSeries[t - numPrev: t + numNext] \
                    .mean(axis=0, keepdims=True)

            idx += 1

        return maSeries

    @staticmethod
    def doubleMovingAverage(
            timeSeries: np.ndarray,
            order1: int,
            order2: int
    ) -> np.ndarray:
        """
        Takes two, moving averages, one after the other, the first one with
        order 'order1' and the second one with order 'order2'. These
        must both be even or odd.

        :param timeSeries: time series whose double-moving average is to be
        estimated. It is a numpy array of shape (n, d)
        :param order1: order of the first moving average taken
        :param order2: order of the second moving average taken
        :return: double-moving averaged time series, it has shape
        (n - order1 - order2 + 2, d)
        """

        assert (isOdd(order1) and isOdd(order2)) \
               or (not isOdd(order1) and not isOdd(order2))

        return MovingAverage.movingAverage(MovingAverage.movingAverage(
            timeSeries, order1, prevMore=False
        ), order2, prevMore=True)
