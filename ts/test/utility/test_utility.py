import pytest
import numpy as np
from ts.utility import Utility


@pytest.mark.parametrize('data, seqLength', [
    (np.linspace(0, 100, 5000), 500),
    (np.linspace(0, 1000, 4350), 1500),
    (np.linspace(0, 100, 1000), 1),
    (np.linspace(0, 1000, 10000), 111),
    (np.random.uniform(0, 100, size=(10000, 20)), 111),
    (np.random.uniform(0, 100, size=(450, 5)), 1)
], ids=['1dim-0', '1dim-1', '1dim-2', '1dim-3', '2dim-0', '2dim-1'])
def test_breakSeq(data: np.ndarray, seqLength: int):
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


def test_trainTestSplit():
    pass


def test_trainTestSplitSeries():
    pass


def test_prepareDataPred():
    pass


def test_prepareDataTrain():
    pass


def test_isExoShapeValid():
    pass


def test_generateMultipleSequence():
    pass
