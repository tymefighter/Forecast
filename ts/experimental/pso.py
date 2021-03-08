import numpy as np


class Pso:

    @staticmethod
    def computeInitialPos(
            paramRange,
            numParticles
    ):
        """

        :param paramRange:
        :param numParticles:
        :return:
        """

        dim = len(paramRange)
        initialPos = np.zeros((numParticles, dim))

        for d in range(dim):
            low, high = paramRange[d]
            initialPos[:, d] = np.random.uniform(low, high, (numParticles,))

        return initialPos

    @staticmethod
    def pso(
            minFunc,
            initialPos,
            inertiaCoeff=1,
            inertiaDamp=0.99,
            personalCoeff=2,
            socialCoeff=2,
            numIterations=20
    ):
        """
        Particle Swarm Optimization algorithm

        :param minFunc: function which is to be minimized. It must
        accept a numpy array of shape (dim,) where dim is the
        dimension of the parameter space
        :param initialPos: initial positions for each of the particles.
        It should be a numpy array of shape (numParticles, dim)
        :param inertiaCoeff: coefficient used for updating the velocity
        based on previous velocity
        :param inertiaDamp: used for damping inertia coefficient after
        every iteration.
        :param personalCoeff: coefficient used for updating the velocity
        based on personal best
        :param socialCoeff: coefficient used for updating the velocity
        based on global best
        :param numIterations: number of iterations to be performed
        :return: (optimized parameters,
            optimal value of the function,
            global best cost values at each iteration),
        where the optimized parameters is a numpy array of shape (dim,)
        and optimal value is the value of the function achieved by these
        parameters
        """

        numParticles, dim = initialPos.shape

        pos = initialPos.copy()
        vel = np.zeros(pos.shape)

        bestPos = initialPos.copy()
        bestCosts = np.zeros((numParticles,))
        bestParticle = None

        for i in range(numParticles):
            bestCosts[i] = minFunc(pos[i])
            bestParticle = i if bestParticle is None \
                or bestCosts[i] < bestCosts[bestParticle] else bestParticle

        iterBestCosts = np.zeros((numIterations,))
        for iterNum in range(numIterations):

            vel = inertiaCoeff * vel \
                + personalCoeff * np.random.rand(numParticles, dim) * (bestPos - pos) \
                + (socialCoeff * np.random.rand(numParticles, dim)
                    * (np.expand_dims(bestPos[bestParticle], axis=0) - pos))

            pos = pos + vel

            for i in range(numParticles):
                currValue = minFunc(pos[i])

                if currValue < bestCosts[i]:
                    bestCosts[i] = currValue
                    bestPos[i, :] = pos[i, :]

                    if currValue < bestCosts[bestParticle]:
                        bestParticle = i

            iterBestCosts[iterNum] = bestCosts[bestParticle]
            inertiaCoeff *= inertiaDamp

        return bestPos[bestParticle], bestCosts[bestParticle], iterBestCosts

    @staticmethod
    def psoConstrictionCoeff(
            minFunc,
            initialPos,
            kappa=1,
            phi1=2.05,
            phi2=2.05,
            numIterations=20
    ):
        """

        :param minFunc:
        :param initialPos:
        :param kappa:
        :param phi1:
        :param phi2:
        :param numIterations:
        :return:
        """

        phi = phi1 + phi2
        chi = 2 * kappa / np.abs(2 - phi - np.sqrt(phi * phi - 4 * phi))

        inertiaCoeff = chi
        inertiaDamp = 1
        personalCoeff = chi * phi1
        socialCoeff = chi * phi2

        return Pso.pso(
            minFunc,
            initialPos,
            inertiaCoeff,
            inertiaDamp,
            personalCoeff,
            socialCoeff,
            numIterations
        )


