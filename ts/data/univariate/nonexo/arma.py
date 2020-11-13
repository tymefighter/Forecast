import numpy as np

from ts.data.univariate.nonexo.non_exo import UnivariateNonExogenous
from ts.log import GlobalLogger


class ArmaGenerator(UnivariateNonExogenous):
    """ARMA as a Univariate Non Exogenous Time Series Data Generator"""

    def __init__(
            self,
            obsCoef,
            noiseCoef,
            noiseGenFunc,
            noiseGenParams,
            obsFunc=None,
            noiseFunc=None
    ):
        """
        Initialize Data Generator

        :param obsCoef: Observation Coefficients, it is a numpy array of shape (p,)
        :param noiseCoef: Noise Coefficients, it is a numpy array of shape (q,)
        :param noiseGenFunc: Noise Generation function (eg. np.random.normal)
        :param noiseGenParams: Parameters to be passed to the Noise Generation Function
        :param obsFunc: Function to be applied on the previous observation terms while computing
        current observation. It may add non-linearity to the time series.
        :param noiseFunc: Function to be applied on the previous noise terms while computing
        current observation. It may add non-linearity to the time series.
        more information
        """

        logger = GlobalLogger.getLogger()
        logger.log('Initialize ARMA coefficients', 1, self.__init__.__name__)

        logger.log(
            f'Shapes - obsCoef: {obsCoef.shape}, noiseCoef: {noiseCoef.shape}',
            2,
            self.__init__.__name__
        )
        logger.log(
            f'Noise Func: {noiseGenFunc.__name__}, Params: {noiseGenParams}',
            2,
            self.__init__.__name__
        )

        assert(len(obsCoef.shape) == 1 and len(noiseCoef.shape) == 1)

        self.obsCoef = obsCoef
        self.noiseCoef = noiseCoef
        self.noiseGenFunc = noiseGenFunc
        self.noiseGenParams = noiseGenParams
        self.obsFunc = obsFunc
        self.noiseFunc = noiseFunc

    def generate(self, n):
        """Generates Sequence of the Provided Length"""

        logger = GlobalLogger.getLogger()
        logger.log(f'Generating Data of length {n}', 1, self.generate.__name__)

        p = self.obsCoef.shape[0]
        q = self.noiseCoef.shape[0]

        x = np.zeros(n)
        eps = np.zeros(n)

        for t in range(n):

            obsVal = 0
            for i in range(min(t, p)):
                obsVal += self.obsCoef[i] * x[t - i - 1]

            if self.obsFunc is not None:
                obsVal = self.obsFunc(obsVal)
            x[t] += obsVal

            noiseVal = 0
            for j in range(min(t, q)):
                noiseVal += self.noiseCoef[j] * eps[t - j - 1]

            if self.noiseFunc is not None:
                noiseVal = self.noiseFunc(noiseVal)
            x[t] += noiseVal

            eps[t] = self.noiseGenFunc(*self.noiseGenParams)
            x[t] += eps[t]

        return x
