import pickle
import numpy as np
from hmmlearn import hmm


class GmmHmmForecast:

    @staticmethod
    def load(modelLoadPath):
        """
        Loads the model from the provided filepath

        :param modelLoadPath: path from where to load the model
        :return: model which is loaded from the given path
        """

        model = GmmHmmForecast(
            None, None, None,
            loadFromFile=True
        )

        with open(modelLoadPath, 'rb') as fl:
            loadDict = pickle.load(fl)

        model.model = loadDict['model']
        model.dimension = loadDict['dimension']
        model.d = loadDict['d']

        return model

    def __init__(
            self,
            numStates,
            numMixtureComp,
            dimension,
            d=10,
            numIterations=10,
            threshold=1e-7,
            covariance_type='full',
            verbose=False,
            loadFromFile=False
    ):
        """
        Initialize GMM-HMM model using the provided parameters, note that
        the training information has to be provided during the now itself,
        and the train function then has to be called exactly once

        :param numStates: number of states of the HMM
        :param numMixtureComp: number of mixture components present in
        each of the emission probabilities
        :param dimension: dimension of the observations
        :param d: number of timesteps to use for forecasting the next
        observation at the next timestep
        :param numIterations: number of iterations of training to be
        performed
        :param threshold: value such that the training procedure
        is said to have converged if the increase in log likelihood
        is lesser than this value
        :param covariance_type: type of covariance matrix to use
        for the mixture components of the GMM emission distribution
        :param verbose: if True, then display training info, else
        do not display training info (training info: log likelihood
        at each iteration)
        :param loadFromFile: True or False - do not use this parameter !,
        this is for internal use only (i.e. it is an implementation detail)
        """

        if loadFromFile:
            self.model = self.dimension = self.d = None
            return

        self.model = hmm.GMMHMM(
            n_components=numStates,
            n_mix=numMixtureComp,
            covariance_type=covariance_type,
            n_iter=numIterations,
            tol=threshold,
            verbose=verbose
        )

        self.dimension = dimension
        self.d = d

    def train(self, trainSequences):
        """
        Train the model on the provided training sequences. This
        function is to be called exactly once.

        :param trainSequences: list of numpy arrays of shape (ni, dimension),
        where each numpy array can represents an observation sequence. Hence,
        each numpy array can have any length (axis 0) but has to have exactly
        'dimension' as the dimension of axis 1
        :return: list of log likelihood values corresponding to each iteration
        """

        X = np.concatenate(trainSequences, axis=0)
        assert X.shape[1] == self.dimension
        lengths = [seq.shape[0] for seq in trainSequences]

        self.model.fit(X=X, lengths=lengths)

        return list(self.model.monitor_.history)

    def predict(self, X, discParamSet, returnMaxLikelihood=False):
        """
        Forecast using the model parameters on the provided input data. A thing
        to note is that only the last prediction is useful, since one already
        has the true observations for the predictions made by this function, but
        this allows one to see the performance of this algorithm.

        :param X: observation sequence, it is a numpy array of shape (n, dimension)
        :param discParamSet: for each of the 'dimension' number of components in
        an observation, there should be a numpy array of containing the discrete
        set of allowed values for that value i.e. for the ith component of the
        observation vector, discParamSet[i] is a numpy array containing all the
        values from which we predict this ith component.
        :param returnMaxLikelihood: if True, our return value is
        (pred, maxLikelihoodValues), i.e. all the predictions, and the log
        likelihood of each prediction. if False, then our return value is
        just the predictions numpy array.
        :return: prediction of the value following every 'd' length
        contiguous subsequence of the provided observation sequence. Hence,
        the predictions numpy array 'pred' has shape (n - d, dimension).
        """

        assert X.shape[1] == self.dimension
        assert len(discParamSet) == self.dimension

        pred = []
        maxLikelihoodValues = [] if returnMaxLikelihood else None

        for t in range(self.d, X.shape[0]):
            x = np.concatenate(
                (X[t - self.d: t], np.zeros((1, self.dimension))),
                axis=0
            )

            obs, maxLikelihood = self.getMostLikelyObs(x, discParamSet, 0)
            pred.append(obs)

            if returnMaxLikelihood:
                maxLikelihoodValues.append(maxLikelihood)

        pred = np.array(pred)

        if returnMaxLikelihood:
            return pred, maxLikelihoodValues
        else:
            return pred

    def save(self, modelSavePath):
        """
        Save the model at the provided filepath

        :param modelSavePath: path where to store the model
        """

        saveDict = {
            'model': self.model,
            'dimension': self.dimension,
            'd': self.d
        }

        with open(modelSavePath, 'wb') as fl:
            pickle.dump(saveDict, fl)

    def getMostLikelyObs(self, x, discParamSet, idx):
        """
        This is a helper function (do not directly call it!). This
        function assumes that among the 'dimension' many components
        of the dth (0-based) observation in x, idx many components have
        been given their values, and all the previous (0..d-1)
        observation vectors in x have also been given their value, now
        we want to compute the most optimal set of values for observation
        components idx..dimension-1, by most optimal we mean values
        for idx..dimension-1 component of x[d] which gives highest log
        likelihood value of the sequence x(0)..x(d).


        :param x: observation sequence, it is a numpy array of shape
        (d + 1, dimension), we want to set the most optimal value of
        x[d + 1, idx : dimension]
        :param discParamSet: for each of the 'dimension' number of components in
        an observation, there should be a numpy array of containing the discrete
        set of allowed values for that value i.e. for the ith component of the
        observation vector, discParamSet[i] is a numpy array containing all the
        values from which we predict this ith component.
        :param idx: index of the component of x[d] whose value we are going to
        fix now, and recursively fix values of components idx+1 .. dimension-1
        such that they yield maximum likelihood
        :return: a 2-tuple containing (chosen observation, likelihood of the
        chosen observation), the chosen observation has maximum likelihood
        given that we are allowed to vary ony these: x[d + 1, idx : dimension]
        (rest all are fixed)
        """

        if idx == len(discParamSet):
            return x[self.d].copy(), self.model.score(x)

        chosenObs = None
        maxLikelihood = None

        for paramValue in discParamSet[idx]:
            x[self.d, idx] = paramValue

            currObs, currLikelihood = \
                self.getMostLikelyObs(x, discParamSet, idx + 1)

            if maxLikelihood is None or currLikelihood > maxLikelihood:
                maxLikelihood = currLikelihood
                chosenObs = currObs

        return chosenObs, maxLikelihood
