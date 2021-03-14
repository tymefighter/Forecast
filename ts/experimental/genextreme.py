import numpy as np
from scipy.stats import genextreme
from ts.experimental.pso import Pso
from ts.experimental.grad_descent_line_search import GradDescentLineSearch


class GeneralizedExtremeValueDistribution:

    def __init__(self, shapeParam, locParam, scaleParam):
        """
        Creates instance of the Generalized Extreme Value Distribution
        based on the shape, location and scale parameters provided

        :param shapeParam: shape parameter of the distribution
        :param locParam: location parameter of the distribution
        :param scaleParam: scale parameter of the distribution
        """

        self.shapeParam = shapeParam
        self.locParam = locParam
        self.scaleParam = scaleParam

    def sample(self, sampleShape):
        """
        Sample from the distribution

        :param sampleShape: shape of the sample
        :return: data sampled from the distribution, it is a numpy array
        of shape 'sampleShape'
        """

        # scipy.stats.genextreme's sampling method 'rvs seems to have a bug:
        # it sometimes outputs points which have 0 probability of occurrence,
        # I sample those values until none of the generated point have 0 prob
        # of occurrence
        data = genextreme.rvs(
            c=-self.shapeParam, loc=self.locParam, scale=self.scaleParam,
            size=sampleShape
        )

        checkTrue = 1 + self.shapeParam * (data - self.locParam) / self.scaleParam <= 0

        while np.any(checkTrue):
            idx = np.where(checkTrue)
            data[idx] = genextreme.rvs(
                c=-self.shapeParam, loc=self.locParam, scale=self.scaleParam,
                size=data[idx].shape
            )

            checkTrue = 1 + self.shapeParam * (data - self.locParam) / self.scaleParam <= 0

        return data

    @staticmethod
    def logLikelihood(shapeParam, locParam, scaleParam, data):
        """
        Computes log likelihood of the data given the parameters

        :param shapeParam: shape parameter of the distribution
        :param locParam: location parameter of the distribution
        :param scaleParam: scale parameter of the distribution
        :param data: data whose log likelihood is to be computed
        :return: log likelihood of the data given the parameters
        """

        if scaleParam <= 0:
            return None

        n = data.shape[0]
        shiftData = (data - locParam) / scaleParam

        if shapeParam == 0:
            return -n * np.log(scaleParam) - np.sum(
                shiftData - np.exp(-shiftData),
                axis=0
            )

        logArg = 1 + shapeParam * shiftData
        if np.any(logArg <= 0):
            return None

        return -n * np.log(scaleParam) - np.sum(
            (1. / shapeParam + 1) * np.log(logArg) + logArg ** (-1.0 / shapeParam),
            axis=0
        )

    @staticmethod
    def computeNegLogLikelihoodGrad(shapeParam, locParam, scaleParam, data):
        """
        Computes the gradient of the log likelihood of the GEV distribution
        with respect to the shape and scale parameters

        :param shapeParam: shape parameter
        :param locParam: location parameter
        :param scaleParam: scale parameter
        :param data: the data, a numpy array of shape (n,)
        :return: (derivative with respect to shape parameter,
            derivative with respect to location parameter
            derivative with respect to scale parameter)
        """

        n = data.shape[0]
        shiftData = (data - locParam) / scaleParam

        if shapeParam == 0:

            shapeGrad = 0

            locGrad = (np.sum(np.exp(-shiftData), axis=0) - n) / scaleParam

            scaleGrad = n / scaleParam - np.sum(
                (data - locParam) * (1 - np.exp(-shiftData)),
                axis=0
            ) / (scaleParam * scaleParam)

        else:

            logArg = 1 + shapeParam * shiftData

            shapeGrad = np.sum(
                - np.log(logArg) / np.square(shapeParam)
                + (1. / shapeParam + 1) * shiftData / logArg
                + (logArg ** (-1. / shapeParam)) * (
                    np.log(logArg) / np.square(shapeParam)
                    - shiftData / (shapeParam * logArg)
                ),
                axis=0
            )

            locGrad = np.sum(
                (shapeParam + 1 - logArg ** (-1. / shapeParam)) / logArg,
                axis=0
            ) / scaleParam

            scaleGrad = n / scaleParam - np.sum(
                ((data - locParam) / logArg) * (
                    shapeParam + 1 - logArg ** (-1. / shapeParam)
                ),
                axis=0
            ) / (scaleParam * scaleParam)

        return shapeGrad, locGrad, scaleGrad


class GevEstimate:

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
        PSO method for maximum likelihood estimation of GEV's parameters

        :param data: the data, it is a numpy array of shape (n,)
        :param initialPos: initial positions of the particles in the
        parameter space, it is a numpy array of shape (numParticles, 3)
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
        where the estimated parameters is a numpy array of shape (3,)
        containing the shape and scale parameters in that order
        """

        def minFunc(param):
            """ The function to minimize - negative log likelihood
             of the data given the parameters """

            logLikelihood = GeneralizedExtremeValueDistribution\
                .logLikelihood(param[0], param[1], param[2], data)

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
            initLocParam,
            initScaleParam,
            learningRate=1e-3,
            learningRateMul=0.80,
            numIterations=10
    ):
        """
        Gradient Descent Line Search based method for maximum likelihood
        estimation of GEV's parameters

        :param data: the data, it is a numpy array of shape (n,)
        :param initShapeParam: initial shape parameter
        :param initLocParam: initial location parameter
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

            negLogLikelihood = GeneralizedExtremeValueDistribution\
                .logLikelihood(param[0], param[1], param[2], data)

            return -negLogLikelihood if negLogLikelihood is not None else None

        def gradFunc(param):
            """ Computes the gradient of negative log likelihood
            of the data given the parameters """

            return np.array(
                GeneralizedExtremeValueDistribution.computeNegLogLikelihoodGrad(
                    param[0], param[1], param[2], data
                ))

        return GradDescentLineSearch.gradDescentLineSearch(
            minFunc, gradFunc, np.array([initShapeParam, initLocParam, initScaleParam]),
            learningRate, learningRateMul, numIterations
        )
