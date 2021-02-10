import pytest
import os
import numpy as np
from ts.model import DeepNN

FILE_PATH = 'model/scratch/model'
EPSILON = 1e-7


@pytest.mark.parametrize('trainSequences, testData, containsExo', [
    (
        [np.random.uniform(-1, 1, size=(length, 4))
         for length in list(np.random.randint(40, 60, size=(5,)))],
        np.random.uniform(-1, 1, size=(50, 4)),
        False
    ),
    (
        [(
            np.random.uniform(-1, 1, size=(length + 1, 3)),
            np.random.uniform(-1, 1, size=(length, 5))
        ) for length in list(np.random.randint(40, 60, size=(5,)))],
        (
            np.random.uniform(-1, 1, size=(50, 3)),
            np.random.uniform(-1, 1, size=(50, 5))
        ), True
    )
], ids=['nonexo', 'exo'])
def test_saveLoad(trainSequences, testData, containsExo):
    """ Test the Save-Load functionality of this model """

    if not containsExo:
        numTargetVariables = testData.shape[1]
        numExoVariables = 0
    else:
        numTargetVariables = testData[0].shape[1]
        numExoVariables = testData[1].shape[1]

    model = DeepNN(
        numTargetVariables=numTargetVariables,
        numExoVariables=numExoVariables,
    )
    model.train(trainSequences)

    if containsExo:
        targetSeries, exogenousSeries = testData
    else:
        targetSeries = testData
        exogenousSeries = None

    predBeforeSave = model.predict(targetSeries, exogenousSeries)

    model.save(FILE_PATH)
    del model

    model = DeepNN.load(FILE_PATH)
    predAfterSave = model.predict(targetSeries, exogenousSeries)

    assert np.array_equal(predBeforeSave, predAfterSave)

    os.remove(FILE_PATH)


def test_checkShapeValid():
    pass


def test_prepareDataPredDNN():
    pass


def test_prepareDataTrainDNN():
    pass


@pytest.mark.parametrize('Ytrue, Ypred', [
    (np.random.uniform(-100, 100, size=(100,)),
        np.random.uniform(-100, 100, size=(100,))),

    (np.random.uniform(-100, 100, size=(40, 3)),
        np.random.uniform(-100, 100, size=(40, 3))),

    (np.random.uniform(-100, 100, size=(15, 5, 5)),
        np.random.uniform(-100, 100, size=(15, 5, 5)))
], ids=['data-0', 'data-1', 'data-2'])
def test_lossFunc(Ytrue, Ypred):

    assert abs(
        DeepNN.lossFunc(Ytrue, Ypred).numpy()
        - np.mean(np.square(Ytrue - Ypred))) < EPSILON


def test_DnnDataSequence():
    pass


def test_predEvalOutputShape():
    """ Test the prediction and evaluation output shape of this model """
    pass
