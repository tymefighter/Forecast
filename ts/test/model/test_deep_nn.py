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

@pytest.mark.parametrize(
    'numTargetVariables, numExoVariables, \
    targetSeries, exogenousSeries, \
    areShapesValid', [
        (14, 0,
            np.random.uniform(-1, 1, size=(100, 3)), None,
            False),
        (3, 5,
            np.random.uniform(-1, 1, size=(100, 3)), None,
            False),
        (3, 0,
            np.random.uniform(-1, 1, size=(100, 3)), None,
            True),
        (3, 0,
            np.random.uniform(-1, 1, size=(100, 3, 1)), None,
            False),
        (3, 2,
            np.random.uniform(-1, 1, size=(100, 3)),
            np.random.uniform(-1, 1, size=(150, 2)),
            True),
        (3, 2,
            np.random.uniform(-1, 1, size=(100, 3)),
            np.random.uniform(-1, 1, size=(150, 2, 4)),
            False),
        (3, 2,
            np.random.uniform(-1, 1, size=(100, 3, 3)),
            np.random.uniform(-1, 1, size=(150, 2)),
            False)
    ], ids=[
        'nonexo-0', 'nonexo-1', 'nonexo-2', 'nonexo-3',
        'exo-0', 'exo-1', 'exo-2'
    ])
def test_checkShapeValid(
    numTargetVariables, numExoVariables,
    targetSeries, exogenousSeries,
    areShapesValid
):
    """ Tests the checkShapeValid method of a DeepNN object """

    model = DeepNN(
        numTargetVariables=numTargetVariables,
        numExoVariables=numExoVariables
    )

    assert areShapesValid == \
           model.checkShapeValid(targetSeries, exogenousSeries)


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


@pytest.mark.parametrize(
    'forecastHorizon, lag, \
    trainSequences, \
    numTargetVariables, numExoVariables, \
    targetTest, exoTest', [
        (
            12, 50,
            [np.random.uniform(-10, 10, size=(length, 10))
             for length in list(np.random.randint(100, 150, size=(5,)))],
            10, 0,
            np.random.uniform(-10, 10, size=(100, 10)), None
        ),
        (
            15, 30,
            [(
                np.random.uniform(-10, 10, size=(length + 15, 10)),
                np.random.uniform(-10, 10, size=(length, 3))
            ) for length in list(np.random.randint(120, 150, size=(3,)))],
            10, 3,
            np.random.uniform(-10, 10, size=(100, 10)),
            np.random.uniform(-10, 10, size=(100, 3))
        )
    ], ids=['nonexo', 'exo'])
def test_predOutputShape(
    forecastHorizon, lag,
    trainSequences,
    numTargetVariables, numExoVariables,
    targetTest, exoTest
):
    """
    Test the prediction and evaluation output shape of this model

    :param forecastHorizon: forecast Horizon
    :param lag: lag parameter for DNN Forecasting model
    :param trainSequences: training Sequences for the model
    :param numTargetVariables: number of target variables
    :param numExoVariables: number of exogenous variables
    :param targetTest: test target series
    :param exoTest: test exogenous series (can be None)
    """

    model = DeepNN(
        forecastHorizon=forecastHorizon,
        lag=lag,
        numTargetVariables=numTargetVariables,
        numExoVariables=numExoVariables
    )

    model.train(trainSequences, returnLosses=False)

    pred = model.predict(targetTest, exoTest)
    assert pred.shape \
           == (targetTest.shape[0] - lag, targetTest.shape[1])


@pytest.mark.parametrize(
    'forecastHorizon, lag, \
    trainSequences, \
    numTargetVariables, numExoVariables, \
    targetEval, exoEval', [
        (
            12, 50,
            [np.random.uniform(-10, 10, size=(length, 10))
             for length in list(np.random.randint(100, 150, size=(5,)))],
            10, 0,
            np.random.uniform(-10, 10, size=(100, 10)), None
        ),
        (
            15, 30,
            [(
                np.random.uniform(-10, 10, size=(length + 15, 10)),
                np.random.uniform(-10, 10, size=(length, 3))
            ) for length in list(np.random.randint(120, 150, size=(3,)))],
            10, 3,
            np.random.uniform(-10, 10, size=(115, 10)),
            np.random.uniform(-10, 10, size=(100, 3))
        )
    ], ids=['nonexo', 'exo'])
def test_evalOutputShape(
    forecastHorizon, lag,
    trainSequences,
    numTargetVariables, numExoVariables,
    targetEval, exoEval
):
    """
    Test the prediction and evaluation output shape of this model

    :param forecastHorizon: forecast Horizon
    :param lag: lag parameter for DNN Forecasting model
    :param trainSequences: training Sequences for the model
    :param numTargetVariables: number of target variables
    :param numExoVariables: number of exogenous variables
    :param targetEval: eval target series
    :param exoEval: eval exogenous series (can be None)
    """

    model = DeepNN(
        forecastHorizon=forecastHorizon,
        lag=lag,
        numTargetVariables=numTargetVariables,
        numExoVariables=numExoVariables
    )

    model.train(trainSequences, returnLosses=False)

    _, evalOut = model.evaluate(targetEval, exoEval, returnPred=True)
    assert evalOut.shape \
           == (targetEval.shape[0] - forecastHorizon - lag, targetEval.shape[1])
