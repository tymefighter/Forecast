import pytest
import os
import numpy as np
from numpy.random import uniform, rand, randint
from ts.model import LstmForecast

FILE_PATH = 'model/scratch/model'


@pytest.mark.parametrize('trainSequences, testData, containsExo', [
    (
        [rand(length, 4) for length in list(randint(40, 60, size=(5,)))],
        rand(50, 4), False
    ),
    (
        [(rand(length + 1, 3), rand(length, 5))
         for length in list(randint(40, 60, size=(5,)))],
        (rand(50, 3), rand(50, 5)), True
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

    model = LstmForecast(
        numTargetVariables=numTargetVariables,
        numExoVariables=numExoVariables
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

    model = LstmForecast.load(FILE_PATH)
    predAfterSave = model.predict(targetSeries, exogenousSeries)

    assert np.array_equal(predBeforeSave, predAfterSave)

    os.remove(FILE_PATH)


@pytest.mark.parametrize(
    'forecastHorizon, \
    trainSequences, \
    numTargetVariables, numExoVariables, \
    targetTest, exoTest', [
        (
            12,
            [uniform(-10, 10, size=(length, 10))
             for length in list(randint(100, 150, size=(5,)))],
            10, 0,
            uniform(-10, 10, size=(100, 10)), None
        ),
        (
            15,
            [(
                uniform(-10, 10, size=(length + 15, 10)),
                uniform(-10, 10, size=(length, 3))
            ) for length in list(randint(120, 150, size=(3,)))],
            10, 3,
            uniform(-10, 10, size=(100, 10)),
            uniform(-10, 10, size=(100, 3))
        )
    ], ids=['nonexo', 'exo'])
def test_predOutputShape(
    forecastHorizon,
    trainSequences,
    numTargetVariables, numExoVariables,
    targetTest, exoTest
):
    """
    Test the prediction and evaluation output shape of this model

    :param forecastHorizon: forecast Horizon
    :param trainSequences: training Sequences for the model
    :param numTargetVariables: number of target variables
    :param numExoVariables: number of exogenous variables
    :param targetTest: test target series
    :param exoTest: test exogenous series (can be None)
    """

    model = LstmForecast(
        forecastHorizon=forecastHorizon,
        numTargetVariables=numTargetVariables,
        numExoVariables=numExoVariables
    )

    model.train(trainSequences, returnLosses=False)

    pred = model.predict(targetTest, exoTest)
    assert pred.shape == targetTest.shape


@pytest.mark.parametrize(
    'forecastHorizon, \
    trainSequences, \
    numTargetVariables, numExoVariables, \
    targetEval, exoEval', [
        (
            12,
            [rand(length, 10) for length in list(randint(100, 150, size=(5,)))],
            10, 0,
            rand(100, 10), None
        ),
        (
            15,
            [(rand(length + 15, 10), rand(length, 3))
             for length in list(randint(120, 150, size=(3,)))],
            10, 3,
            rand(115, 10),
            rand(100, 3)
        )
    ], ids=['nonexo', 'exo'])
def test_evalOutputShape(
    forecastHorizon,
    trainSequences,
    numTargetVariables, numExoVariables,
    targetEval, exoEval
):
    """
    Test the prediction and evaluation output shape of this model

    :param forecastHorizon: forecast Horizon
    :param trainSequences: training Sequences for the model
    :param numTargetVariables: number of target variables
    :param numExoVariables: number of exogenous variables
    :param targetEval: eval target series
    :param exoEval: eval exogenous series (can be None)
    """

    model = LstmForecast(
        forecastHorizon=forecastHorizon,
        numTargetVariables=numTargetVariables,
        numExoVariables=numExoVariables
    )

    model.train(trainSequences, returnLosses=False)

    _, evalOut = model.evaluate(targetEval, exoEval, returnPred=True)
    assert evalOut.shape \
           == (targetEval.shape[0] - forecastHorizon, targetEval.shape[1])
