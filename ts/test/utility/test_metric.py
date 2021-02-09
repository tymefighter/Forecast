import pytest
import numpy as np
from ts.utility import Metric


def sampleAboutZero(n: int, d: int, minAbs: float, maxAbs: float):
    """
    Samples an array of specified number of samples and dimensions,
    with each value being independently sampled. The elements of
    this array can be give negative or positive but not zero

    :param n: number of samples
    :param d: number of dimensions
    :param minAbs: abs of minimum value allowed
    :param maxAbs: abs of maximum value allowed
    :return: numpy array of shape (n, d), where each element is
    randomly sampled, and has value in range
    [-maxAbs, -minAbs) U [minAbs, maxAbs)
    """
    arr = np.zeros((n, d))

    for i in range(n):
        for j in range(d):
            if np.random.rand() < 0.5:
                arr[i, j] = np.random.uniform(-maxAbs, -minAbs)
            else:
                arr[i, j] = np.random.uniform(minAbs, maxAbs)

    return arr


@pytest.mark.parametrize('forecastSeq, actualSeq', [
    (
        np.random.uniform(0.1, 1000, (100, 1)),
        np.random.uniform(0.1, 1000, (100, 1))
    ),
    (
        np.random.uniform(0.1, 1000, (500, 4)),
        np.random.uniform(0.1, 1000, (500, 4))
    ),
    (
        np.random.uniform(-10, 0, (155, 12)),
        np.random.uniform(-10, 0, (155, 12))
    ),
    (
        sampleAboutZero(20, 10, 1e-4, 1.0),
        sampleAboutZero(20, 10, 1e-4, 1.0)
    )
], ids=['data-pos-0', 'data-pos-1', 'data-neg-0', 'data-mix-0'])
def test_mape(forecastSeq: np.ndarray, actualSeq: np.ndarray):
    """
    Tests the MAPE metric
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

    :param forecastSeq: Forecast (Predicted) sequence, a numpy array
    of shape (n, d)
    :param actualSeq: Actual sequence, a numpy array of shape (n, d)
    """

    assert forecastSeq.shape == actualSeq.shape
    assert len(forecastSeq.shape) == 2

    mape = Metric.mape(forecastSeq, actualSeq)
    assert len(mape.shape) == 1 and mape.shape[0] == forecastSeq.shape[1]

    assert np.array_equal(
        mape,
        100. * np.mean(np.abs((actualSeq - forecastSeq) / actualSeq), axis=0)
    )


@pytest.mark.parametrize('forecastSeq, actualSeq', [
    (
        np.random.uniform(-1000, 1000, (100, 1)),
        np.random.uniform(-1000, 1000, (100, 1))
    ),
    (
        np.random.uniform(-100, 100, (500, 4)),
        np.random.uniform(-100, 100, (500, 4))
    ),
    (
        np.random.uniform(-1e-3, 1e-3, (155, 12)),
        np.random.uniform(-1e-3, 1e-3, (155, 12))
    ),
    (
        np.random.uniform(-20, 20, (155, 12)),
        np.random.uniform(-20, 20, (155, 12))
    ),
], ids=['data-0', 'data-1', 'data-2', 'data-3'])
def test_mae(forecastSeq: np.ndarray, actualSeq: np.ndarray):
    """
    Tests the MAE metric
    https://en.wikipedia.org/wiki/Mean_absolute_error

    :param forecastSeq: Forecast (Predicted) sequence, a numpy array
    of shape (n, d)
    :param actualSeq: Actual sequence, a numpy array of shape (n, d)
    """

    assert forecastSeq.shape == actualSeq.shape
    assert len(forecastSeq.shape) == 2

    mae = Metric.mae(forecastSeq, actualSeq)
    assert len(mae.shape) == 1 and mae.shape[0] == forecastSeq.shape[1]

    assert np.array_equal(mae, np.mean(np.abs(actualSeq - forecastSeq), axis=0))


@pytest.mark.parametrize('forecastSeq, actualSeq', [
    (
        np.random.uniform(0.1, 1000, (100, 1)),
        np.random.uniform(0.1, 1000, (100, 1))
    ),
    (
        np.random.uniform(0.1, 1000, (500, 4)),
        np.random.uniform(0.1, 1000, (500, 4))
    ),
    (
        np.random.uniform(-10, 0, (155, 12)),
        np.random.uniform(-10, 0, (155, 12))
    ),
    (
        sampleAboutZero(20, 10, 1e-4, 1.0),
        sampleAboutZero(20, 10, 1e-4, 1.0)
    )
], ids=['data-pos-0', 'data-pos-1', 'data-neg-0', 'data-mix-0'])
def test_mpe(forecastSeq: np.ndarray, actualSeq: np.ndarray):
    """
    Tests the MPE metric
    https://en.wikipedia.org/wiki/Mean_percentage_error

    :param forecastSeq: Forecast (Predicted) sequence, a numpy array
    of shape (n, d)
    :param actualSeq: Actual sequence, a numpy array of shape (n, d)
    """

    assert forecastSeq.shape == actualSeq.shape
    assert len(forecastSeq.shape) == 2

    mpe = Metric.mpe(forecastSeq, actualSeq)
    assert len(mpe.shape) == 1 and mpe.shape[0] == forecastSeq.shape[1]

    assert np.array_equal(
        mpe,
        100. * np.mean((actualSeq - forecastSeq) / actualSeq, axis=0)
    )


@pytest.mark.parametrize('forecastSeq, actualSeq', [
    (
        np.random.uniform(-1000, 1000, (100, 1)),
        np.random.uniform(-1000, 1000, (100, 1))
    ),
    (
        np.random.uniform(-100, 100, (500, 4)),
        np.random.uniform(-100, 100, (500, 4))
    ),
    (
        np.random.uniform(-1e-3, 1e-3, (155, 12)),
        np.random.uniform(-1e-3, 1e-3, (155, 12))
    ),
    (
        np.random.uniform(-20, 20, (155, 12)),
        np.random.uniform(-20, 20, (155, 12))
    ),
], ids=['data-0', 'data-1', 'data-2', 'data-3'])
def test_mse(forecastSeq: np.ndarray, actualSeq: np.ndarray):
    """
    Tests the MSE metric
    https://en.wikipedia.org/wiki/Mean_squared_error

    :param forecastSeq: Forecast (Predicted) sequence, a numpy array
    of shape (n, d)
    :param actualSeq: Actual sequence, a numpy array of shape (n, d)
    """

    assert forecastSeq.shape == actualSeq.shape
    assert len(forecastSeq.shape) == 2

    mse = Metric.mse(forecastSeq, actualSeq)
    assert len(mse.shape) == 1 and mse.shape[0] == forecastSeq.shape[1]

    assert np.array_equal(mse, np.mean(np.square(actualSeq - forecastSeq), axis=0))


@pytest.mark.parametrize('forecastSeq, actualSeq', [
    (
        np.random.uniform(-1000, 1000, (100, 1)),
        np.random.uniform(-1000, 1000, (100, 1))
    ),
    (
        np.random.uniform(-100, 100, (500, 4)),
        np.random.uniform(-100, 100, (500, 4))
    ),
    (
        np.random.uniform(-1e-3, 1e-3, (155, 12)),
        np.random.uniform(-1e-3, 1e-3, (155, 12))
    ),
    (
        np.random.uniform(-20, 20, (155, 12)),
        np.random.uniform(-20, 20, (155, 12))
    ),
], ids=['data-0', 'data-1', 'data-2', 'data-3'])
def test_mse(forecastSeq: np.ndarray, actualSeq: np.ndarray):
    """
    Tests the RMSE metric
    https://en.wikipedia.org/wiki/Root-mean-square_deviation

    :param forecastSeq: Forecast (Predicted) sequence, a numpy array
    of shape (n, d)
    :param actualSeq: Actual sequence, a numpy array of shape (n, d)
    """

    assert forecastSeq.shape == actualSeq.shape
    assert len(forecastSeq.shape) == 2

    rmse = Metric.rmse(forecastSeq, actualSeq)
    assert len(rmse.shape) == 1 and rmse.shape[0] == forecastSeq.shape[1]

    assert np.array_equal(
        rmse,
        np.sqrt(np.mean(np.square(actualSeq - forecastSeq), axis=0))
    )
