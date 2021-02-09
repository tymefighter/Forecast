import pytest
import numpy as np
from ts.utility import Utility
from ts.data.generate.univariate.nonexo \
    import StandardGenerator, PolynomialGenerator, DifficultGenerator


@pytest.mark.parametrize('data, seqLength', [
    (np.linspace(0, 100, 5000), 500),
    (np.linspace(0, 1000, 4350), 1500),
    (np.linspace(0, 100, 1000), 1),
    (np.linspace(0, 1000, 10000), 111),
    (np.random.uniform(0, 100, size=(10000, 20)), 111),
    (np.random.uniform(0, 100, size=(450, 5)), 1)
], ids=['1dim-0', '1dim-1', '1dim-2', '1dim-3', '2dim-0', '2dim-1'])
def test_breakSeq(data: np.ndarray, seqLength: int):
    """ Tests Utility.breakSeq """

    dataSeq = Utility.breakSeq(data, seqLength)

    # On concatenating the dataSeq, we should get back data
    assert np.array_equal(np.concatenate(dataSeq, axis=0), data)

    # length of each seq except the last should be exactly seqLength
    for seq in dataSeq[:-1]:
        assert seq.shape[0] == seqLength


@pytest.mark.parametrize(
    'targetSeries, exogenousSeries, seqLength, forecastHorizon', [
        (np.linspace(0, 100, 5000), None, 300, None),
        (np.random.uniform(0, 1000, size=(500, 20)), None, 1000, None),
        (np.random.uniform(0, 100, size=(10000, 20)), None, 111, None),
        (
            np.random.uniform(0, 100, size=(10000, 7)),
            np.random.uniform(0, 100, size=(10000, 4)),
            350, 1
        ),
        (
            np.random.uniform(0, 1000, size=(5000,)),
            np.random.uniform(0, 400, size=(5000, 10)),
            735, 15
        ),
        (
            np.random.uniform(0, 200, size=(1000, 8)),
            np.random.uniform(0, 200, size=(1000, 4)),
            1500, 11
        )
    ], ids=['nonexo-0', 'nonexo-1', 'nonexo-2', 'exo-0', 'exo-1', 'exo-2'])
def test_breakTrainSeq(
        targetSeries,
        exogenousSeries,
        seqLength,
        forecastHorizon
):
    """ Tests Utility.breakTrainSeq """

    n = targetSeries.shape[0]

    trainSequences = Utility.breakTrainSeq(
        targetSeries, exogenousSeries, seqLength, forecastHorizon
    )

    # If exogenousSeries is none, breakTrainSeq behaves differently,
    # here we test that behaviour
    if exogenousSeries is None:
        # On concatenating the trainSequences, we should get back data
        assert np.array_equal(np.concatenate(trainSequences, axis=0), targetSeries)

        # length of each seq except the last should be exactly seqLength
        for seq in trainSequences[:-1]:
            assert seq.shape[0] == seqLength

        return

    # Forecast horizon cannot be None
    assert forecastHorizon is not None

    # Target and Exogenous series must have same number of elements
    assert targetSeries.shape[0] == exogenousSeries.shape[0]

    # Check if the train sequences are correct
    startIdx = 0
    for (targetSeriesSeq, exogenousSeriesSeq) in trainSequences:
        lenTargetSeries = targetSeriesSeq.shape[0]
        lenExogenousSeries = exogenousSeriesSeq.shape[0]

        assert lenTargetSeries == lenExogenousSeries + forecastHorizon

        exoEndIdx = startIdx + lenExogenousSeries
        targetEndIdx = exoEndIdx + forecastHorizon

        # Check if the broken sequence matches the correct part of the
        # original sequence
        assert np.array_equal(
            targetSeriesSeq, targetSeries[startIdx:targetEndIdx]
        )
        assert np.array_equal(
            exogenousSeriesSeq,
            exogenousSeries[startIdx:exoEndIdx]
        )

        startIdx = exoEndIdx

    assert (startIdx + forecastHorizon == n) or (n - startIdx <= forecastHorizon)


@pytest.mark.parametrize('dataSequences, containsExo, forecastHorizon', [
    (
        [np.random.uniform(0, 100, size=(100,)) for i in range(50)],
        False, 10
    ),
    (
        [np.random.uniform(0, 100, size=(n,))
            for n in list(np.random.randint(0, 100, size=50))],
        False, 50
    ),
    (
        [np.random.uniform(0, 100, size=(n, 20))
            for n in list(np.random.randint(0, 100, size=50))],
        False, 50
    ),
    (
        [
            (np.random.uniform(0, 100, size=(n,)),
             np.random.uniform(0, 100, size=(n, 5)))
            for n in list(np.random.randint(0, 100, size=50))
        ],
        True, 50
    ),
    (
        [
            (np.random.uniform(0, 100, size=(n, 4)),
             np.random.uniform(0, 100, size=(n, 5)))
            for n in list(np.random.randint(0, 100, size=50))
        ],
        True, 50
    )
], ids=['nonexo-0', 'nonexo-1', 'nonexo-2', 'exo-0', 'exo-1'])
def test_convertToTrainSeq(dataSequences, containsExo, forecastHorizon):
    """ Tests Utility.convertToTrainSeq """

    trainSequences = \
        Utility.convertToTrainSeq(dataSequences, containsExo, forecastHorizon)

    dataSeqIdx = 0
    trainSeqIdx = 0
    while dataSeqIdx < len(dataSequences):
        if containsExo:

            (dataTarget, dataExo) = dataSequences[dataSeqIdx]
            dataLen = dataTarget.shape[0]
            if dataLen > forecastHorizon:
                (trainTarget, trainExo) = trainSequences[trainSeqIdx]

                assert np.array_equal(trainTarget, dataTarget)
                assert np.array_equal(trainExo, dataExo[:dataLen - forecastHorizon])
                trainSeqIdx += 1

        else:

            dataTarget = dataSequences[dataSeqIdx]
            dataLen = dataTarget.shape[0]
            if dataLen > forecastHorizon:
                trainTarget = trainSequences[trainSeqIdx]

                assert np.array_equal(trainTarget, dataTarget)
                trainSeqIdx += 1

        dataSeqIdx += 1

    assert trainSeqIdx == len(trainSequences)


@pytest.mark.parametrize(
    'data, train, val', [
        (np.random.uniform(0, 1000, size=(5000,)), 3500, None),
        (np.random.uniform(0, 1000, size=(5000, 10)), 4900, None),
        (np.random.uniform(0, 1000, size=(100, 4)), 1, None),
        (np.random.uniform(0, 1000, size=(5000, 4)), 0.9, None),
        (np.random.uniform(0, 1000, size=(2550, 4)), 0.7, None),
        (np.random.uniform(0, 1000, size=(3000, 4)), 0.1, None),

        (np.random.uniform(0, 1000, size=(5000,)), 3500, 500),
        (np.random.uniform(0, 1000, size=(5000, 10)), 4900, 95),
        (np.random.uniform(0, 1000, size=(100, 4)), 1, 1),
        (np.random.uniform(0, 1000, size=(100, 4)), 1, 0.4),
        (np.random.uniform(0, 1000, size=(5000, 4)), 0.9, 0.05),
        (np.random.uniform(0, 1000, size=(2550, 4)), 0.7, 0.1),
        (np.random.uniform(0, 1000, size=(2550, 4)), 0.7, 0.0),
    ], ids=[
        'no_val-0', 'no_val-1', 'no_val-2', 'no_val-3', 'no_val-4', 'no_val-5',
        'val-0', 'val-1', 'val-2', 'val-3', 'val-4', 'val-5', 'val-6'
    ])
def test_trainTestSplit(data, train, val):
    """ Tests Utility.trainTestSplit """

    if train < 1.0:
        train = round(data.shape[0] * train)

    # If validation set is not required
    if val is None:
        dataTrain, dataTest = Utility.trainTestSplit(data, train, None)

        # train and test data together should give entire data
        assert np.array_equal(np.concatenate((dataTrain, dataTest), axis=0), data)

        # Train data must have the required number of elements
        assert dataTrain.shape[0] == train

        return

    dataTrain, dataVal, dataTest = Utility.trainTestSplit(data, train, val)

    # train, val and test data together should give entire data
    assert np.array_equal(
        np.concatenate((dataTrain, dataVal, dataTest), axis=0), data
    )

    if val < 1.0:
        val = round(data.shape[0] * val)

    # Train and Val data must have the required number of elements
    assert dataTrain.shape[0] == train
    assert dataVal.shape[0] == val


@pytest.mark.parametrize('targetSeries, exogenousSeries, train, val', [
    (
        np.random.uniform(0, 1000, size=(5000,)),
        np.random.uniform(0, 1000, size=(5000, 1)),
        4500, None
    ),
    (
        np.random.uniform(0, 2000, size=(5000, 10)),
        np.random.uniform(0, 2000, size=(5000, 5)),
        4999, None
    ),
    (
        np.random.uniform(0, 2000, size=(100, 7)),
        np.random.uniform(0, 2000, size=(100, 3)),
        1, 1
    ),
    (
        np.random.uniform(0, 2000, size=(100, 3)),
        np.random.uniform(0, 2000, size=(100, 7)),
        98, 1
    ),
    (
        np.random.uniform(0, 2000, size=(1000, 4)),
        np.random.uniform(0, 2000, size=(1000, 6)),
        0.7, 0.2
    ),
    (
        np.random.uniform(0, 2000, size=(1000, 5)),
        np.random.uniform(0, 2000, size=(1000, 5)),
        0.5, 0.1
    ),
], ids=['no_val-0', 'no_val-1', 'val-0', 'val-1', 'val-2', 'val-3'])
def test_trainTestSplitSeries(targetSeries, exogenousSeries, train, val):
    """ Tests Utility.trainTestSplitSeries """

    assert targetSeries.shape[0] == exogenousSeries.shape[0]
    n = targetSeries.shape[0]

    if train < 1.0:
        train = round(n * train)

    # If validation set is not required
    if val is None:
        (targetTrain, exoTrain), (targetTest, exoTest) = \
            Utility.trainTestSplitSeries(targetSeries, exogenousSeries, train, None)

        # train and test data together should give entire data
        assert np.array_equal(
            np.concatenate((targetTrain, targetTest), axis=0), targetSeries
        )
        assert np.array_equal(
            np.concatenate((exoTrain, exoTest), axis=0), exogenousSeries
        )

        assert targetTrain.shape[0] == exoTrain.shape[0] == train

        return

    if val < 1.0:
        val = round(n * val)

    (targetTrain, exoTrain), (targetVal, exoVal), (targetTest, exoTest) = \
        Utility.trainTestSplitSeries(targetSeries, exogenousSeries, train, val)

    # train, val and test data together should give entire data
    assert np.array_equal(
        np.concatenate((targetTrain, targetVal, targetTest), axis=0), targetSeries
    )
    assert np.array_equal(
        np.concatenate((exoTrain, exoVal, exoTest), axis=0), exogenousSeries
    )

    assert targetTrain.shape[0] == exoTrain.shape[0] == train
    assert targetVal.shape[0] == exoVal.shape[0] == val


@pytest.mark.parametrize('targetSeries, exogenousSeries', [
    (np.random.uniform(0, 100, size=(100,)), None),
    (np.random.uniform(0, 100, size=(100, 4)), None),
    (
        np.random.uniform(0, 100, size=(50,)),
        np.random.uniform(0, 100, size=(50, 4))
    ),
    (
        np.random.uniform(0, 100, size=(50, 1)),
        np.random.uniform(0, 100, size=(50, 1))
    ),
    (
        np.random.uniform(0, 100, size=(50, 3)),
        np.random.uniform(0, 100, size=(50, 4))
    )
], ids=['nonexo-0', 'nonexo-1', 'exo-0', 'exo-1', 'exo-2'])
def test_prepareDataPred(targetSeries, exogenousSeries):
    """ Tests Utility.prepareDataPred """

    if exogenousSeries is not None:
        assert targetSeries.shape[0] == exogenousSeries.shape[0]

    n = targetSeries.shape[0]
    X = Utility.prepareDataPred(targetSeries, exogenousSeries)

    if len(targetSeries.shape) == 1:
        d1 = 1
    else:
        d1 = targetSeries.shape[1]

    if exogenousSeries is None:
        d2 = 0
    else:
        d2 = exogenousSeries.shape[1]

    # Shape of features must equal to (n, d1 + d2)
    assert X.shape == (n, d1 + d2)

    for i in range(n):
        x = targetSeries[i]
        if not isinstance(x, np.ndarray):
            x = np.array([x])

        if exogenousSeries is not None:
            x = np.concatenate((x, exogenousSeries[i]), axis=0)

        # Concatenation of ith target and exo elements must be
        # the ith element of features X
        assert np.array_equal(x, X[i])


@pytest.mark.parametrize('targetSeries, exogenousSeries, forecastHorizon', [
    (np.random.uniform(0, 100, size=(151,)), None, 1),
    (np.random.uniform(0, 100, size=(110,)), None, 10),
    (np.random.uniform(0, 100, size=(104, 5)), None, 4),
    (
        np.random.uniform(0, 100, size=(112,)),
        np.random.uniform(0, 100, size=(100, 3)),
        12
    ),
    (
        np.random.uniform(0, 100, size=(115, 5)),
        np.random.uniform(0, 100, size=(100, 3)),
        15
    ),
    (
        np.random.uniform(0, 100, size=(101, 5)),
        np.random.uniform(0, 100, size=(100, 5)),
        1
    ),
    (
        np.random.uniform(0, 100, size=(130, 2)),
        np.random.uniform(0, 100, size=(100, 5)),
        30
    )
])
def test_prepareDataTrain(targetSeries, exogenousSeries, forecastHorizon):
    """ Tests Utility.prepareDataTrain """

    if exogenousSeries is not None:
        assert targetSeries.shape[0] == exogenousSeries.shape[0] + forecastHorizon

    assert targetSeries.shape[0] > forecastHorizon
    n = targetSeries.shape[0] - forecastHorizon

    X, Y = Utility.prepareDataTrain(targetSeries, exogenousSeries, forecastHorizon)
    assert n == X.shape[0] == Y.shape[0]  # Shapes of features and targets must agree

    # The targets must equal to the target series starting at 'forecastHorizon'
    assert np.array_equal(Y, targetSeries[forecastHorizon:])

    xConstruct = targetSeries[:n]
    if len(xConstruct.shape) == 1:
        xConstruct = np.expand_dims(xConstruct, axis=1)

    if exogenousSeries is not None:
        xConstruct = np.concatenate((xConstruct, exogenousSeries), axis=1)

    # Features must match the concatenation of target series and exo series
    assert np.array_equal(X, xConstruct)


@pytest.mark.parametrize('exogenousSeries, numExoVariables, isValid', [
    (None, 0, True),
    (np.random.uniform(0, 10, size=(10, 1)), 1, True),
    (np.random.uniform(0, 10, size=(20, 4)), 4, True),
    (np.random.uniform(0, 10, size=(15, 15)), 15, True),

    (None, 1, False),
    (None, 5, False),
    (np.random.uniform(0, 10, size=(20,)), 1, False),
    (np.random.uniform(0, 10, size=(20, 4)), 3, False),
    (np.random.uniform(0, 10, size=(20, 4, 2)), 4, False)
], ids=[
    'valid-0', 'valid-1', 'valid-2', 'valid-3',
    'invalid-0', 'invalid-1', 'invalid-2', 'invalid-3', 'invalid-4'
])
def test_isExoShapeValid(exogenousSeries, numExoVariables, isValid):
    """ Tests Utility.isExoShapeValid """

    assert Utility.isExoShapeValid(exogenousSeries, numExoVariables) == isValid


@pytest.mark.parametrize(
    'dataGenerator, numSequences, minSequenceLength, maxSequenceLength', [
        (StandardGenerator(), 100, 20, 20),
        (StandardGenerator(), 200, 1, 200),
        (PolynomialGenerator(), 30, 40, 42),
        (PolynomialGenerator(), 20, 20, 20),
        (DifficultGenerator(), 1, 1, 1),
        (DifficultGenerator(), 2, 1, 2)
    ], ids=['gen-0', 'gen-1', 'gen-2', 'gen-3', 'gen-4', 'gen-5'])
def test_generateMultipleSequence(
    dataGenerator,
    numSequences,
    minSequenceLength,
    maxSequenceLength
):
    """ Tests Utility.generateMultipleSequence """

    dataSeq = Utility.generateMultipleSequence(
        dataGenerator,
        numSequences,
        minSequenceLength,
        maxSequenceLength
    )

    assert len(dataSeq) == numSequences

    for seq in dataSeq:
        assert minSequenceLength <= seq.shape[0] <= maxSequenceLength
