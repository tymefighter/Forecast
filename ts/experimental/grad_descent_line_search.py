import numpy as np


class GradDescentLineSearch:

    @staticmethod
    def gradDescentLineSearch(
            minFunc,
            gradFunc,
            initParam,
            learningRate=1e-3,
            learningRateMul=0.80,
            numIterations=10
    ):
        """
        Gradient Descent Line Search based method for minimizing
        the provided function

        :param minFunc: function which is to be minimized
        :param gradFunc: function which computes gradient of minFunc
        with respect to the parameters at a given point in the parameter
        space
        :param initParam: initial parameter values
        :param learningRate: learning rate
        :param learningRateMul: factor with which to multiply the
        learning rate if parameters go out of the parameter domain
        :param numIterations: number of iterations to be performed
        :return: (estimated parameters, negative log likelihood over
        each iteration)
        """

        param = initParam
        assert minFunc(param) is not None

        minFuncIters = np.zeros((numIterations,))

        for iterNum in range(numIterations):
            paramGrad = gradFunc(param)

            rate = learningRate
            currParam = param - rate * paramGrad

            currMinFuncValue = minFunc(currParam)
            while currMinFuncValue is None:
                rate *= learningRateMul
                currParam = param - rate * paramGrad
                currMinFuncValue = minFunc(currParam)

            param = currParam
            minFuncIters[iterNum] = currMinFuncValue

        return param, minFuncIters
