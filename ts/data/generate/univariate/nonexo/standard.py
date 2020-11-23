import numpy as np

from ts.data.generate.univariate.nonexo.non_exo import UnivariateNonExogenous
from ts.data.generate.univariate.nonexo.arma import ArmaGenerator
from ts.log import GlobalLogger


class StandardGenerator(UnivariateNonExogenous):
    """
    Standard Data Generation

    Generates Common Types Required Types of Data, like Simple, Long
    Term and Extreme. It provides slight variants as well.
    """

    def __init__(self, typeOfData='simple'):
        """
        Initialize the Standard Data Generator

        :param typeOfData: Affects only the coefficients of the ARMA data generator.
        Choices: simple, long_term, extreme_short, extreme_long, extreme_different.
        Note that the prefix 'extreme' just refers to the noise being generated from
        an extreme valued distribution (for example Gumbel, Lognormal)
        """
        GlobalLogger.getLogger().log(
            f'Initializing Standard Generator using type: {typeOfData}',
            1,
            self.__init__.__name__
        )

        if typeOfData == 'simple':
            obsCoef = np.array([0.5, -0.2, 0.2])
            noiseCoef = np.array([0.5, -0.2])

            noiseGenFunc = np.random.normal
            noiseGenParams = (10.0, 1.0)

        elif typeOfData == 'long_term':
            obsCoef, noiseCoef = \
                StandardGenerator.generateRandomArmaCoeff(50, 50)

            noiseGenFunc = np.random.normal
            noiseGenParams = (10.0, 1.0)

        elif typeOfData == 'extreme_short':
            obsCoef, noiseCoef = \
                StandardGenerator.generateRandomArmaCoeff(5, 5)

            noiseGenFunc = np.random.lognormal
            noiseGenParams = (1.0, 1.0)

        elif typeOfData == 'extreme_long':
            obsCoef, noiseCoef = \
                StandardGenerator.generateRandomArmaCoeff(50, 50)

            noiseGenFunc = np.random.lognormal
            noiseGenParams = (1.0, 1.0)

        elif typeOfData == 'extreme_different':
            obsCoef, noiseCoef = \
                StandardGenerator.generateRandomArmaCoeff(5, 5)

            noiseGenFunc = np.random.gumbel
            noiseGenParams = (-5., 10.0)

        else:
            raise Exception("Invalid Type of Data")

        self.armaGen = ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams)

    def generate(self, n):
        """Generates Sequence of the Provided Length"""

        return self.armaGen.generate(n)

    @staticmethod
    def generateRandomArmaCoeff(P, Q):
        """
        Generate Random Coefficients for Observation and Noise Terms present in
        the ARMA Generator

        :param P: Length of Observation Coefficients
        :param Q: Length of Noise Coefficients
        :return: Observation Coefficients and Noise Coefficients
        """

        obsCoef = np.concatenate([
            np.random.uniform(-0.1, 0, size=P // 2),
            np.random.uniform(0, 0.1, size=P // 2)
        ])

        noiseCoef = np.concatenate([
            np.random.uniform(-0.01, 0, size=Q // 2),
            np.random.uniform(0, 0.01, size=Q // 2)
        ])

        return obsCoef, noiseCoef
