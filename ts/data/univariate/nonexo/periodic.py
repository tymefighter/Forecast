import numpy as np

from ts.data.univariate.nonexo.non_exo import UnivariateNonExogenous
from ts.log import GlobalLogger


class PeriodicGenerator(UnivariateNonExogenous):
    """Periodic Univariate Non Exogenous Time Series Data Generator"""

    def __init__(
            self,
            initialPhase=0,
            period=5.0
    ):
        """
        Initialize Data Generator for Periodic Generator which generates data
        using the sine function

        :param initialPhase: Initial Phase of sine function
        :param period: Time Period of the sine function
        """

        GlobalLogger\
            .getLogger()\
            .log('Initialize Periodic Generator', 1, self.__init__.__name__)

        self.initialPhase = initialPhase
        self.angFreq = 2 * np.pi / period

    def generate(self, n):
        """Generates Sequence of the Provided Length"""

        logger = GlobalLogger.getLogger()
        logger.log(f'Generating Data of length {n}', 1, self.generate.__name__)

        x = np.zeros(n)

        for t in range(n):
            x[t] = np.sin(self.initialPhase + self.angFreq * t)

        return x
