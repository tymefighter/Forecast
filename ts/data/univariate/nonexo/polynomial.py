import numpy as np

from ts.data.univariate.nonexo.non_exo import UnivariateNonExogenous
from ts.log import GlobalLogger


class PolynomialGenerator(UnivariateNonExogenous):
    """Polynomial Univariate Non Exogenous Time Series Data Generator"""

    def __init__(self, polynomialCoeffs):
        """
        Initialize Data Generator for Polynomial Generator which generates data
        using the provided polynomial coefficient list

        :param polynomialCoeffs: Polynomial Coefficient List which is used to
        generate data which is a polynomial function of time. polynomialCoeffs[i]
        is the coefficient of pow(t, i)
        """

        GlobalLogger\
            .getLogger()\
            .log('Initialize Polynomial Generator', 1, self.__init__.__name__)

        self.polynomialCoeffs = polynomialCoeffs

    def generate(self, n):
        """Generates Sequence of the Provided Length"""

        logger = GlobalLogger.getLogger()
        logger.log(f'Generating Data of length {n}', 1, self.generate.__name__)

        x = np.zeros(n)

        for t in range(n):
            pw = 1
            for coeff in self.polynomialCoeffs:
                x[t] += coeff * pw
                pw *= t

        return x
