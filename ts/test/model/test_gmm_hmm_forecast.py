import pytest
import os
import numpy as np
from ts.model import GmmHmmForecast

FILE_PATH = 'model/scratch/model'


def test_saveLoad():
    """ Test the Save-Load functionality of this model """

    dim = 2
    xTrain = np.random.uniform(-1, 1, size=(100, dim))
    xTest = np.random.uniform(-1, 1, size=(20, dim))
    discParamSet = [np.random.uniform(-1, 1, size=(5,)) for _ in range(dim)]

    model = GmmHmmForecast(2, 2, dim, numIterations=2)
    model.train([xTrain])
    predBeforeSave = model.predict(xTest, discParamSet)

    model.save(FILE_PATH)
    del model

    model = GmmHmmForecast.load(FILE_PATH)
    predAfterSave = model.predict(xTest, discParamSet)

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
    latency = 2
    dim = xTrain.shape[1]

    model = GmmHmmForecast(2, 2, dim, d=latency, numIterations=2)
    model.train([xTrain])

    discParamSet = [np.random.uniform(-1, 1, size=(5,)) for _ in range(dim)]
    pred = model.predict(xTest, discParamSet)

    assert pred.shape == (xTest.shape[0] - latency, dim)
