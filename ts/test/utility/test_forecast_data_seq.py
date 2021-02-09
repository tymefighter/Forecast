import pytest
import numpy as np
from ts.utility import ForecastDataSequence


@pytest.mark.parametrize(
    'trainSequences, forecastHorizon, numTargetVariables, numExoVariables', [
        (
            [np.random.uniform(-10, 10, size=(length, 5))
             for length in list(np.random.randint(50, 100, size=(20,)))],
            10, 5, 0
        ),
        (
            [np.random.uniform(-10, 10, size=(length, 1))
             for length in list(np.random.randint(50, 100, size=(5,)))],
            17, 1, 0
        ),
        (
            [np.random.uniform(-10, 10, size=(length, 14))
             for length in list(np.random.randint(100, 102, size=(5,)))],
            99, 14, 0
        ),
        (
            [(
                np.random.uniform(-10, 10, size=(length + 12, 4)),
                np.random.uniform(-10, 10, size=(length, 5))
              ) for length in list(np.random.randint(50, 100, size=(5,)))],
            12, 4, 5
        ),
        (
            [(
                np.random.uniform(-10, 10, size=(length + 8, 7)),
                np.random.uniform(-10, 10, size=(length, 3))
            ) for length in list(np.random.randint(50, 100, size=(10,)))],
            8, 7, 3
        )
    ], ids=['nonexo-0', 'nonexo-1', 'nonexo-2', 'exo-0', 'exo-1'])
def test_ForecastDataSequence(
    trainSequences,
    forecastHorizon,
    numTargetVariables,
    numExoVariables
):
    """
    Tests the ForecastDataSequence class

    :param trainSequences: Sequences (List) of data, each element in the
    list is a target sequence of shape (n, numTargetVariables) or a tuple
    containing a target sequence of shape (n + forecastHorizon, numTargetVariables)
    and an exogenous sequence of shape (n, numExoVariables)
    :param forecastHorizon: How much further in the future the model has to
    predict the target series variable
    :param numTargetVariables: Number of target variables the model takes as input
    :param numExoVariables: Number of exogenous variables the model takes as input
    """

    forecastDataSequence = ForecastDataSequence(
        trainSequences,
        forecastHorizon,
        numTargetVariables,
        numExoVariables
    )

    assert len(forecastDataSequence) == len(trainSequences)

    n = len(forecastDataSequence)
    for i in range(n):
        X, Y = forecastDataSequence[i]

        assert len(X.shape) == len(Y.shape) == 3
        assert X.shape[0] == Y.shape[0] == 1
        assert X.shape[1] == Y.shape[1]
        assert X.shape[2] == numTargetVariables + numExoVariables
        assert Y.shape[2] == numTargetVariables

        X = np.squeeze(X, axis=0)
        Y = np.squeeze(Y, axis=0)

        if numExoVariables == 0:
            assert not isinstance(trainSequences[i], tuple)
            assert np.array_equal(trainSequences[i][:-forecastHorizon], X)
            assert np.array_equal(trainSequences[i][forecastHorizon:], Y)

        else:
            assert isinstance(trainSequences[i], tuple)
            assert len(trainSequences[i]) == 2
            assert trainSequences[i][0].shape[0] \
                   == trainSequences[i][1].shape[0] + forecastHorizon

            assert np.array_equal(
                np.concatenate((
                    trainSequences[i][0][:-forecastHorizon],
                    trainSequences[i][1]
                ), axis=1),
                X
            )
            assert np.array_equal(trainSequences[i][0][forecastHorizon:], Y)
