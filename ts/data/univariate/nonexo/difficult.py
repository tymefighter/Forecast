from ts.data.univariate.nonexo.non_exo import UnivariateNonExogenous
from ts.data.univariate.nonexo.standard import StandardGenerator
from ts.data.univariate.nonexo.periodic import PeriodicGenerator
from ts.data.univariate.nonexo.polynomial import PolynomialGenerator
from ts.log import GlobalLogger


class DifficultGenerator(UnivariateNonExogenous):
    """
    Difficult Data Generator

    The generated data has a period (seasonality), a trend and noise
    """

    def __init__(
            self,
            typeOfData='simple',
            periodType='long',
            trendType='linear_inc',
            biasType='none'
    ):
        """
        Initialize the Difficult Data Generator

        :param typeOfData: Affects only the coefficients of the ARMA data generator.
        Choices: simple, long_term, extreme_short, extreme_long, extreme_different.
        Note that the prefix 'extreme' just refers to the noise being generated from
        an extreme valued distribution (for example Gumbel, Lognormal)
        :param periodType: Type of period the data should follow. Choices: long, short
        :param trendType: Type of trend the data should follow. Choices: linear_inc,
        linear_dec, poly_inc, poly_dec
        :param biasType: Type of bias the data should have. Choices: none, pos, neg.
        Note that the 'none' bias type is a string and NOT the Python None
        """

        GlobalLogger.getLogger().log(
            f'Type of data: {typeOfData}, period type: {periodType}, '
            + f'trend type: {trendType}, bias type: {biasType}',
            1,
            self.__init__.__name__
        )

        self.standardGen = StandardGenerator(typeOfData)

        if periodType == 'long':
            period = 10.0
        elif periodType == 'short':
            period = 1.0
        else:
            raise Exception("Invalid Period Type")

        if trendType == 'linear_inc':
            coeff = [0.0, 1.0]
        elif trendType == 'linear_dec':
            coeff = [0.0, -1.0]
        elif trendType == 'poly_inc':
            coeff = [0.0, 0.2, 0.5]
        elif trendType == 'poly_dec':
            coeff = [0.0, -0.2, -0.5]
        else:
            raise Exception("Invalid Trend Type")

        if biasType == 'none':
            coeff[0] = 0
        elif biasType == 'pos':
            coeff[0] = 5.0
        elif biasType == 'neg':
            coeff[0] = -5.0
        else:
            raise Exception("Invalid Bias Type")

        self.periodicGen = PeriodicGenerator(period=period)
        self.polyGen = PolynomialGenerator(coeff)

    def generate(self, n):
        """Generates Sequence of the Provided Length"""

        return \
            self.standardGen.generate(n) + \
            self.periodicGen.generate(n) + \
            self.polyGen.generate(n)
