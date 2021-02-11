import pytest
import os
import numpy as np
from numpy.random import uniform, rand, randint
from ts.model import DeepNN
from ts.model.deep_nn import DnnDataSequence

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


@pytest.mark.parametrize('targetSeries, exogenousSeries, lag', [
    (rand(115, 12), None, 15),
    (rand(234, 5), None, 24),
    (rand(117, 4), rand(117, 7), 17),
    (rand(85, 11), rand(85, 7), 35),
], ids=['nonexo-0', 'nonexo-1', 'exo-0', 'exo-1'])
def test_prepareDataPredDNN(targetSeries, exogenousSeries, lag):
    """
    Tests 'prepareDataPredDNN' static method of DeepNN class

    :param targetSeries: target series
    :param exogenousSeries: exogenous series (can be None)
    :param lag: number of timesteps of the past we should
    use for prediction (excluding current timestep)
    """

    if exogenousSeries is not None:
        assert targetSeries.shape[0] == exogenousSeries.shape[0]

    n = targetSeries.shape[0]
    numInputs = n - lag
    assert numInputs > 0

    numTargetVariables = targetSeries.shape[1]
    numExoVariables = 0 if exogenousSeries is None else exogenousSeries.shape[1]

    X = DeepNN.prepareDataPredDNN(targetSeries, exogenousSeries, lag)
    assert X.shape == (numInputs, (numTargetVariables + numExoVariables) * (lag + 1))

    for i in range(lag, n):

        x = []
        for j in range(i - lag, i + 1):
            x.append(targetSeries[j])
            if exogenousSeries is not None:
                x.append(exogenousSeries[j])

        x = np.concatenate(x, axis=0)

        assert np.array_equal(x, X[i - lag])


@pytest.mark.parametrize('targetSeries, exogenousSeries, forecastHorizon, lag', [
    (rand(119, 12), None, 4, 15),
    (rand(241, 5), None, 7, 24),
    (rand(128, 4), rand(117, 7), 11, 17),
    (rand(95, 11), rand(85, 7), 10, 35),
], ids=['nonexo-0', 'nonexo-1', 'exo-0', 'exo-1'])
def test_prepareDataTrainDNN(
    targetSeries, exogenousSeries,
    forecastHorizon, lag
):
    """
    Tests 'prepareDataTrainDNN' static method of DeepNN class

    :param targetSeries: target series
    :param exogenousSeries: exogenous series (can be None)
    :param forecastHorizon: how much ahead in the future we
    should predict
    :param lag: number of timesteps of the past we should
    use for prediction (excluding current timestep)
    """

    if exogenousSeries is not None:
        assert targetSeries.shape[0] \
               == exogenousSeries.shape[0] + forecastHorizon

    n = targetSeries.shape[0]
    numInputs = n - lag - forecastHorizon
    assert numInputs > 0

    numTargetVariables = targetSeries.shape[1]
    numExoVariables = 0 if exogenousSeries is None else exogenousSeries.shape[1]

    X, Y = DeepNN.prepareDataTrainDNN(
        targetSeries, exogenousSeries, forecastHorizon, lag
    )

    assert X.shape[0] == Y.shape[0] == numInputs
    assert X.shape[1] == (numTargetVariables + numExoVariables) * (lag + 1) \
        and Y.shape[1] == numTargetVariables

    for i in range(lag, lag + numInputs):
        x = []
        for j in range(i - lag, i + 1):
            x.append(targetSeries[j])
            if exogenousSeries is not None:
                x.append(exogenousSeries[j])

        x = np.concatenate(x, axis=0)
        y = targetSeries[i + forecastHorizon]

        assert np.array_equal(x, X[i - lag])
        assert np.array_equal(y, Y[i - lag])


@pytest.mark.parametrize('Ytrue, Ypred', [
    (uniform(-100, 100, size=(100,)),
        uniform(-100, 100, size=(100,))),

    (uniform(-100, 100, size=(40, 3)),
        uniform(-100, 100, size=(40, 3))),

    (uniform(-100, 100, size=(15, 5, 5)),
        uniform(-100, 100, size=(15, 5, 5)))
], ids=['data-0', 'data-1', 'data-2'])
def test_lossFunc(Ytrue, Ypred):
    """ Test the loss function of DeepNN """

    assert abs(
        DeepNN.lossFunc(Ytrue, Ypred).numpy()
        - np.mean(np.square(Ytrue - Ypred))) < EPSILON


@pytest.mark.parametrize(
    'trainSequences, forecastHorizon, \
    numTargetVariables, numExoVariables, lag', [
        ([rand(length + 33, 4)
          for length in list(randint(100, 120, size=(5,)))],
         33, 4, 0, 12),
        ([rand(length + 12, 5)
          for length in list(randint(60, 70, size=(1,)))],
         12, 5, 0, 3),
        ([(rand(length + 4, 2), rand(length, 3))
          for length in list(randint(60, 70, size=(9,)))],
         4, 2, 3, 10),
        ([(rand(length + 14, 5), rand(length, 5))
          for length in list(randint(60, 70, size=(8,)))],
         14, 5, 5, 5)
    ], ids=['nonexo-0', 'nonexo-1', 'exo-0', 'exo-1'])
def test_DnnDataSequence(
    trainSequences, forecastHorizon,
    numTargetVariables, numExoVariables, lag
):
    """
    Test DnnDataSequence class

    :param trainSequences: training sequences
    :param forecastHorizon: forecast horizon
    :param numTargetVariables: number of target variables
    :param numExoVariables: number of exogenous variables
    :param lag: lag value
    """

    dnnDataSequence = DnnDataSequence(
        trainSequences, forecastHorizon,
        numTargetVariables, numExoVariables, lag
    )

    assert len(dnnDataSequence) == len(trainSequences)
    n = len(dnnDataSequence)

    for idx in range(n):
        seq = trainSequences[idx]
        if numExoVariables == 0:
            targetSeries = seq
            exogenousSeries = None
        else:
            targetSeries = seq[0]
            exogenousSeries = seq[1]

        n = targetSeries.shape[0]
        numInputs = n - lag - forecastHorizon

        X, Y = dnnDataSequence[idx]
        assert X.shape[0] == Y.shape[0] == numInputs
        assert X.shape[1] == (numTargetVariables + numExoVariables) * (lag + 1) \
            and Y.shape[1] == numTargetVariables

        for i in range(lag, lag + numInputs):
            x = []
            for j in range(i - lag, i + 1):
                x.append(targetSeries[j])
                if exogenousSeries is not None:
                    x.append(exogenousSeries[j])

            x = np.concatenate(x, axis=0)
            y = targetSeries[i + forecastHorizon]

            assert np.array_equal(x, X[i - lag])
            assert np.array_equal(y, Y[i - lag])


@pytest.mark.parametrize(
    'forecastHorizon, lag, \
    trainSequences, \
    numTargetVariables, numExoVariables, \
    targetTest, exoTest', [
        (
            12, 50,
            [uniform(-10, 10, size=(length, 10))
             for length in list(randint(100, 150, size=(5,)))],
            10, 0,
            uniform(-10, 10, size=(100, 10)), None
        ),
        (
            15, 30,
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
