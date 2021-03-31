import numpy as np
from scipy.stats import genpareto
from ts.experimental.pso import Pso
from ts.experimental.grad_descent_line_search import GradDescentLineSearch


class GeneralizedParetoDistribution:

    def __init__(self, shapeParam, scaleParam):
        """
        Creates instance of the Generalized Pareto Distribution
        based on the shape and scale parameters provided

        :param shapeParam: shape parameter of the distribution
        :param scaleParam: scale parameter of the distribution
        """

        self.shapeParam = shapeParam
        self.scaleParam = scaleParam

    def sample(self, sampleShape):
        """
        Sample from the distribution

        :param sampleShape: shape of the sample
        :return: data sampled from the distribution, it is a numpy array
        of shape 'sampleShape'
        """

        return genpareto.rvs(
            c=self.shapeParam, loc=0, scale=self.scaleParam, size=sampleShape
        )

    def computeQuantile(self, p):
        """
        Compute the p-quantile of this distribution

        :param p: CDF probability
        :return: the point z such that CDF(z) = p, i.e. the
        p-quantile of this distribution
        """

        if self.scaleParam != 0:
            return self.scaleParam * ((1 - p) ** (-self.shapeParam) - 1) / self.shapeParam

        else:
            return - self.scaleParam * np.ln(1 - p)

    def pdf(self, x):
        """
        Compute PDF for all values in the input

        :param x: scalar or a numpy array of any shape
        :return: scalar value if x is scalar, or numpy array of shape
        same as x if x is a numpy array. This is the PDF at every point in x
        """

        if self.shapeParam != 0:
            return (1 + self.shapeParam * x / self.scaleParam) ** (-1 / self.shapeParam - 1) \
                / self.scaleParam

        else:
            return np.exp(-x / self.scaleParam) / self.scaleParam

    def cdf(self, x):
        """
        Compute CDF for all values in the input

        :param x: scalar or a numpy array of any shape
        :return: scalar value if x is scalar, or numpy array of shape
        same as x if x is a numpy array. This is the CDF at every point in x
        """

        if self.shapeParam != 0:
            return 1 - (1 + self.shapeParam * x / self.scaleParam) ** (-1 / self.shapeParam)

        else:
            return 1 - np.exp(-x / self.scaleParam)

    @staticmethod
    def logLikelihood(shapeParam, scaleParam, data):
        """
        Computes log likelihood of the data given the parameters

        :param shapeParam: shape parameter of the distribution
        :param scaleParam: scale parameter of the distribution
        :param data: data whose log likelihood is to be computed
        :return: log likelihood of the data given the parameters
        """

        if scaleParam <= 0:
            return None

        n = data.shape[0]

        if shapeParam == 0:
            if np.any(data < 0):
                return None

            return -n * np.log(scaleParam) - np.sum(data, axis=0) / scaleParam

        logArg = 1 + shapeParam * data / scaleParam
        if np.any(logArg <= 0):
            return None

        return -n * np.log(scaleParam) \
               - (1 / shapeParam + 1) * np.sum(np.log(logArg), axis=0)

    @staticmethod
    def computeNegLogLikelihoodGrad(shapeParam, scaleParam, data):
        """
        Computes the gradient of the log likelihood of the GPD distribution
        with respect to the shape and scale parameters

        :param shapeParam: shape parameter
        :param scaleParam: scale parameter
        :param data: the data, a numpy array of shape (n,)
        :return: (derivative with respect to shape parameter,
            derivative with respect to scale parameter)
        """

        n = data.shape[0]

        if shapeParam == 0:
            shapeGrad, scaleGrad = 0, n / scaleParam \
                                   - np.sum(data, axis=0) / (scaleParam * scaleParam)

        else:
            logArg = 1 + shapeParam * data / scaleParam
            shapeGrad = \
                -np.sum(np.log(logArg), axis=0) / np.square(shapeParam) \
                + (1 + 1 / shapeParam) * np.sum(data / logArg, axis=0) / scaleParam

            scaleGrad = n / scaleParam \
                - ((1 + 1 / shapeParam)
                   * shapeParam
                   * (1 / np.square(scaleParam))
                   * np.sum(data / logArg, axis=0))

        return shapeGrad, scaleGrad


class GpdEstimate:

    @staticmethod
    def psoMethod(
            data,
            initialPos,
            inertiaCoeff=1,
            inertiaDamp=0.99,
            personalCoeff=2,
            socialCoeff=2,
            numIterations=20
    ):
        """
        PSO method for maximum likelihood estimation of GPD's parameters

        :param data: the data, it is a numpy array of shape (n,)
        :param initialPos: initial positions of the particles in the
        parameter space, it is a numpy array of shape (numParticles, 2)
        :param inertiaCoeff: coefficient used for updating the velocity
        based on previous velocity
        :param inertiaDamp: used for damping inertia coefficient after
        every iteration.
        :param personalCoeff: coefficient used for updating the velocity
        based on personal best
        :param socialCoeff: coefficient used for updating the velocity
        based on global best
        :param numIterations: number of iterations to be performed
        :return: (estimated parameters,
            maximum value of the log likelihood,
            global maximum likelihood over each iteration),
        where the estimated parameters is a numpy array of shape (2,)
        containing the shape and scale parameters in that order
        """

        def minFunc(param):
            """ The function to minimize - negative log likelihood
             of the data given the parameters """

            logLikelihood = GeneralizedParetoDistribution\
                .logLikelihood(param[0], param[1], data)

            return -logLikelihood if logLikelihood is not None else np.inf

        params, bestCost, bestCosts = Pso.pso(
            minFunc,
            initialPos,
            inertiaCoeff,
            inertiaDamp,
            personalCoeff,
            socialCoeff,
            numIterations
        )

        return params, -bestCost, np.array(list(map(lambda x: -x, list(bestCosts))))

    @staticmethod
    def gradDescentLineSearch(
            data,
            initShapeParam,
            initScaleParam,
            learningRate=1e-3,
            learningRateMul=0.80,
            numIterations=10
    ):
        """
        Gradient Descent Line Search based method for maximum likelihood
        estimation of GPD's parameters

        :param data: the data, it is a numpy array of shape (n,)
        :param initShapeParam: initial shape parameter
        :param initScaleParam: initial scale parameter
        :param learningRate: learning rate
        :param learningRateMul: factor with which to multiply the
        learning rate if parameters go out of the parameter domain
        :param numIterations: number of iterations to be performed
        :return: (estimated parameters, negative log likelihood over
        each iteration)
        """

        def minFunc(param):
            """ The function to minimize - negative log likelihood
             of the data given the parameters """

            negLogLikelihood = GeneralizedParetoDistribution\
                .logLikelihood(param[0], param[1], data)

            return -negLogLikelihood if negLogLikelihood is not None else None

        def gradFunc(param):
            """ Computes the gradient of negative log likelihood
            of the data given the parameters """

            return np.array(GeneralizedParetoDistribution
                            .computeNegLogLikelihoodGrad(param[0], param[1], data))

        return GradDescentLineSearch.gradDescentLineSearch(
            minFunc, gradFunc, np.array([initShapeParam, initScaleParam]),
            learningRate, learningRateMul, numIterations
        )
