import pytest
import numpy as np
from ts.utility import Utility


@pytest.mark.parametrize('data, seqLength', [
    (np.linspace(0, 100, 5000), 500),
    (np.linspace(0, 1000, 4350), 1500),
    (np.linspace(0, 100, 1000), 1500),
    (np.linspace(0, 100, 1000), 1),
    (np.linspace(0, 1000, 10000), 111)
])
def test_breakSeq(data: np.ndarray, seqLength: int):
    dataSeq = Utility.breakSeq(data, seqLength)

    # On concatenating the dataSeq, we should get back data
    assert np.array_equal(np.concatenate(dataSeq, axis=0), data)

    # length of each seq except the last should be exactly seqLength
    for seq in dataSeq[:-1]:
        assert seq.shape[0] == seqLength


def test_breakTrainSeq():
    pass


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
