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
