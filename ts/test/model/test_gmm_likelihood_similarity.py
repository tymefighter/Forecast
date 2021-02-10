import pytest
import os
import numpy as np
from ts.model import GmmHmmLikelihoodSimilarity

FILE_PATH = 'model/scratch/model'


def test_saveLoad():
    """ Test the Save-Load functionality of this model """

    dim = 2
    xTrain = np.random.uniform(-1, 1, size=(100, dim))
    xTest = np.random.uniform(-1, 1, size=(20, dim))

    model = GmmHmmLikelihoodSimilarity(2, 2, dim, numIterations=2)
    model.train([xTrain])
    predBeforeSave = model.predict(xTest)

    model.save(FILE_PATH)
    del model

    model = GmmHmmLikelihoodSimilarity.load(FILE_PATH)
    predAfterSave = model.predict(xTest)

    assert np.array_equal(predBeforeSave, predAfterSave)

    os.remove(FILE_PATH)


@pytest.mark.parametrize('xTrain, xTest', [
    (
        np.random.uniform(-1, 1, size=(50, 2)),
        np.random.uniform(-1, 1, size=(4, 2))
    ),
    (
        np.random.uniform(-1, 1, size=(73, 3)),
        np.random.uniform(-1, 1, size=(11, 3))
    ),
    (
        np.random.uniform(-1, 1, size=(10, 1)),
        np.random.uniform(-1, 1, size=(7, 1))
    )
], ids=['data-0', 'data-1', 'data-2'])
def test_predictOutputShape(xTrain, xTest):
    """ Test the prediction output shape of this model """

    assert xTrain.shape[1] == xTest.shape[1]
    dim = xTrain.shape[1]

    model = GmmHmmLikelihoodSimilarity(2, 2, dim, numIterations=2)
    model.train([xTrain])
    pred = model.predict(xTest)

    assert pred.shape == xTest.shape


def genTest_ClosestLikelihoodObsDiff(
    numTrainSeq, minLengthTrain, maxLengthTrain, dim, numLikelihoodQueries
):
    """
    Generate Test Cases of test_ClosestLikelihoodObsDiff

    :param numTrainSeq:
    :param minLengthTrain:
    :param maxLengthTrain:
    :param dim:
    :param numLikelihoodQueries:
    :return:
    """

    trainSequences = [
        np.random.uniform(-1, 1, size=(length, dim))
        for length in list(np.random.randint(
            minLengthTrain, maxLengthTrain, size=(numTrainSeq,)
        ))
    ]

    model = GmmHmmLikelihoodSimilarity(2, 2, dim, numIterations=2)
    model.train(trainSequences)

    likelihoodObsDiff = model\
        .closestLikelihoodObsDiff\
        .likelihoodObsDiff

    minVal = maxVal = None
    for likelihood, _ in likelihoodObsDiff:
        if minVal is None:
            minVal = maxVal = likelihood
        else:
            minVal = min(minVal, likelihood)
            maxVal = max(maxVal, likelihood)

    likelihoodQueries = np.linspace(minVal, maxVal, numLikelihoodQueries)

    return model, trainSequences, likelihoodQueries


@pytest.mark.parametrize('model, trainSequences, likelihoodQueries', [
    genTest_ClosestLikelihoodObsDiff(4, 20, 40, 10, 20),
    genTest_ClosestLikelihoodObsDiff(3, 20, 30, 14, 25)
], ids=['data-0', 'data-1'])
def test_ClosestLikelihoodObsDiff(model, trainSequences, likelihoodQueries):
    """ Test the 'ClosestLikelihoodObsDiff' Data Structure """

    closestLikelihoodObsDiff = model.closestLikelihoodObsDiff

    for logLikelihood in likelihoodQueries:
        obsDiffClosestLikelihoodObsDiff = closestLikelihoodObsDiff\
            .getClosestLikelihoodObsDiff(logLikelihood)

        obsDiffCalc = None
        minLogLikelihoodDiff = None

        for seq in trainSequences:
            for i in range(seq.shape[0] - 1):
                currObs = seq[i]
                nextObs = seq[i + 1]
                currLogLikelihoodDiff = abs(model.model.score(
                    np.expand_dims(currObs, axis=0)
                ) - logLikelihood)

                if minLogLikelihoodDiff is None \
                        or minLogLikelihoodDiff >= currLogLikelihoodDiff:
                    obsDiffCalc = nextObs - currObs
                    minLogLikelihoodDiff = currLogLikelihoodDiff

        assert np.array_equal(obsDiffClosestLikelihoodObsDiff, obsDiffCalc)
