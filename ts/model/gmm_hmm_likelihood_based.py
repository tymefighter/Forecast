import pickle
import numpy as np
from hmmlearn import hmm


class GmmHmmLikelihoodBased:
    """
    GMM-HMM forecasting model based on the paper:
    "Stock Market Forecasting Using Hidden Markov Model: A New Approach
    Md. Rafiul Hassan and Baikunth Nath, The University of Melbourne,
    Carlton 3010, Australia"
    link:
    http://mleg.cse.sc.edu/edu/csce768/uploads/Main.ReadingList/HMM-stock.pdf
    """

    @staticmethod
    def load(modelLoadPath):
        """
        Loads the model from the provided filepath

        :param modelLoadPath: path from where to load the model
        :return: model which is loaded from the given path
        """

        model = GmmHmmLikelihoodBased(
            None, None, None,
            loadFromFile=True
        )

        with open(modelLoadPath, 'rb') as fl:
            loadDict = pickle.load(fl)

        model.model = loadDict['model']
        model.closestLikelihoodObsDiff = loadDict['closestLikelihoodObsDiff']
        model.dimension = loadDict['dimension']

        return model

    def __init__(
            self,
            numStates,
            numMixtureComp,
            dimension,
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
            self.model = self.dimension = None
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
        self.closestLikelihoodObsDiff = None

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
        self.closestLikelihoodObsDiff = ClosestLikelihoodObsDiff(
            hmmModel=self.model, trainSequences=trainSequences
        )

        return list(self.model.monitor_.history)

    def predict(self, X):
        """
        Forecast using the model parameters on the provided input data. A thing
        to note is that only the last prediction is useful, since one already
        has the true observations for the predictions made by this function, but
        this allows one to see the performance of this algorithm.

        :param X: observation sequence, it is a numpy array of shape (n, dimension)
        :return: for every observation in X, predict the next timestep value, hence
        prediction is a numpy array of shape (n, d)
        """

        pred = []
        for i in range(X.shape[0]):
            currObs = X[i]
            currLogLikelihood = self.model.score(np.expand_dims(currObs, axis=0))
            obsDiff = self\
                .closestLikelihoodObsDiff\
                .getClosestLikelihoodObsDiff(currLogLikelihood)
            nextObs = currObs + obsDiff

            pred.append(nextObs)

        pred = np.array(pred)
        return pred

    def save(self, modelSavePath):
        """
        Save the model at the provided filepath

        :param modelSavePath: path where to store the model
        """

        assert self.closestLikelihoodObsDiff is not None, 'fit should be called'

        saveDict = {
            'model': self.model,
            'closestLikelihoodObsDiff': self.closestLikelihoodObsDiff,
            'dimension': self.dimension
        }

        with open(modelSavePath, 'wb') as fl:
            pickle.dump(saveDict, fl)


class ClosestLikelihoodObsDiff:
    """
    Data Structure for finding the observation in the current dataset
    whose likelihood is closest to the provided input likelihood as a
    query. Actually, instead of outputting the observation, it outputs
    the difference between next and current observation found.
    """

    def __init__(self, hmmModel, trainSequences):
        """
        Constructs the data structure using the trained HMM model and
        training sequences

        :param hmmModel: The trained HMM model
        :param trainSequences: Training Sequences
        """

        self.likelihoodObsDiff = []

        for seq in trainSequences:
            for i in range(seq.shape[0] - 1):
                obsDiff = seq[i + 1] - seq[i]
                logLikelihood = hmmModel.score(np.expand_dims(seq[i], axis=0))

                self.likelihoodObsDiff.append((logLikelihood, obsDiff))

        self.likelihoodObsDiff.sort(
            key=lambda logLikelihoodObsDiff: logLikelihoodObsDiff[0]
        )

    def getClosestLikelihoodObsDiff(self, logLikelihood):
        """
        Outputs the observation difference between the observation
        which has closest log likelihood to 'logLikelihood' and
        the next observation to this observation which is found.
        This is used using the binary search technique

        :param logLikelihood: The input log likelihood
        :return: observation difference between the observation
        which has closest log likelihood to 'logLikelihood' and
        the next observation to this observation which is found
        """

        n = len(self.likelihoodObsDiff)

        if logLikelihood <= self.likelihoodObsDiff[0][0]:
            return self.likelihoodObsDiff[0][1]
        elif logLikelihood >= self.likelihoodObsDiff[n - 1][0]:
            return self.likelihoodObsDiff[n - 1][1]

        low = 0
        high = n - 1
        ansIdx = 0

        # Binary Search - find largest log likelihood which is
        # smaller or equal to the input 'logLikelihood'
        while low <= high:
            mid = (low + high) >> 1

            if self.likelihoodObsDiff[mid][0] <= logLikelihood:
                ansIdx = max(ansIdx, mid)
                low = mid + 1
            else:
                high = mid - 1

        if (logLikelihood - self.likelihoodObsDiff[ansIdx][0]
                < self.likelihoodObsDiff[ansIdx + 1][0] - logLikelihood):
            return self.likelihoodObsDiff[ansIdx][1]

        else:
            return self.likelihoodObsDiff[ansIdx + 1][1]
