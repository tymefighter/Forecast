import numpy as np
from scipy.stats import genpareto
from ts.experimental.pso import Pso


class GeneralizedParetoDistribution:

    def __init__(self, shapeParam, scaleParam):
        """

        :param shapeParam:
        :param scaleParam:
        """

        self.shapeParam = shapeParam
        self.scaleParam = scaleParam

    def sample(self, sampleShape):
        """

        :param sampleShape:
        :return:
        """

        return genpareto.rvs(
            c=self.shapeParam, loc=0, scale=self.scaleParam, size=sampleShape
        )

    def pdf(self, x):
        """

        :param x:
        :return:
        """

        return genpareto.pdf(x, c=self.shapeParam, loc=0, scale=self.scaleParam)

    def cdf(self, x):
        """

        :param x:
        :return:
        """

        return genpareto.cdf(x, c=self.shapeParam, loc=0, scale=self.scaleParam)

    @staticmethod
    def logLikelihood(shapeParam, scaleParam, data):
        """

        :param shapeParam:
        :param scaleParam:
        :param data:
        :return:
        """

        if scaleParam <= 0:
            return None

        n = data.shape[0]

        if shapeParam == 0:
            return -n * np.log(scaleParam) - np.sum(data, axis=0) / scaleParam

        logArg = 1 + shapeParam * data / scaleParam
        if np.any(logArg <= 0):
            return None

        return -n * np.log(scaleParam) \
               - (1 / shapeParam + 1) * np.sum(np.log(logArg), axis=0)

    @staticmethod
    def checkParam(shapeParam, scaleParam, data):

        return scaleParam > 0 and np.all(1 + shapeParam * data / scaleParam > 0)


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

        :param data:
        :param initialPos:
        :param inertiaCoeff:
        :param inertiaDamp:
        :param personalCoeff:
        :param socialCoeff:
        :param numIterations:
        :return:
        """

        def minFunc(param):
            """  """

            shapeParam, scaleParam = param[0], param[1]
            logLikelihood = GeneralizedParetoDistribution\
                .logLikelihood(shapeParam, scaleParam, data)

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
        assert GeneralizedParetoDistribution\
            .checkParam(initShapeParam, initScaleParam, data)

        shapeParam, scaleParam = initShapeParam, initScaleParam
        negLogLikelihoods = np.zeros((numIterations,))

        for iterNum in range(numIterations):
            shapeGrad, scaleGrad = GpdEstimate.computeGrad(shapeParam, scaleParam, data)

            rate = learningRate
            newShapeParam, newScaleParam = shapeParam - rate * shapeGrad,\
                scaleParam - rate * scaleGrad

            while not GeneralizedParetoDistribution\
                    .checkParam(newShapeParam, newScaleParam, data):

                rate *= learningRateMul
                newShapeParam, newScaleParam = shapeParam - rate * shapeGrad, \
                    scaleParam - rate * scaleGrad

            shapeParam = newShapeParam
            scaleParam = newScaleParam

            negLogLikelihoods[iterNum] = -GeneralizedParetoDistribution\
                .logLikelihood(shapeParam, scaleParam, data)

        return np.array([shapeParam, scaleParam]), negLogLikelihoods

    @staticmethod
    def computeGrad(shapeParam, scaleParam, data):
        """  """

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
