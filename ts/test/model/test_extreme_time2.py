import pytest
from numpy.random import rand
from ts.model import ExtremeTime2

FILE_PATH = 'model/scratch/model'


@pytest.mark.parametrize(
    'forecastHorizon, \
    targetSeries, exogenousSeries, seqLength, \
    numExoVariables, \
    targetTest, exoTest', [
        (
            12,
            rand(200), None, 50,
            0,
            rand(100), None
        ),
        (
            15,
            rand(215), rand(200, 3), 50,
            3,
            rand(100), rand(100, 3)
        )
    ], ids=['nonexo', 'exo'])
def test_predOutputShape(
    forecastHorizon,
    targetSeries, exogenousSeries, seqLength,
    numExoVariables,
    targetTest, exoTest
):
    """
    Test the prediction and evaluation output shape of this model

    :param forecastHorizon: forecast Horizon
    :param targetSeries train target series
    :param exogenousSeries train exogenous series
    :param seqLength train sequence length
    :param numExoVariables: number of exogenous variables
    :param targetTest: test target series
    :param exoTest: test exogenous series (can be None)
    """

    model = ExtremeTime2(
        forecastHorizon=forecastHorizon,
        memorySize=5,
        windowSize=5,
        embeddingSize=5,
        contextSize=5,
        numExoVariables=numExoVariables
    )

    model.train(targetSeries, seqLength, exogenousSeries, returnLosses=False)

    pred = model.predict(targetTest, exoTest)
    assert pred.shape == targetTest.shape


@pytest.mark.parametrize(
    'forecastHorizon, \
    targetSeries, exogenousSeries, seqLength, \
    numExoVariables, \
    targetEval, exoEval', [
        (
            12,
            rand(200), None, 50,
            0,
            rand(100), None
        ),
        (
            15,
            rand(215), rand(200, 3), 50,
            3,
            rand(115), rand(100, 3)
        )
    ], ids=['nonexo', 'exo'])
def test_evalOutputShape(
    forecastHorizon,
    targetSeries, exogenousSeries, seqLength,
    numExoVariables,
    targetEval, exoEval
):
    """
    Test the prediction and evaluation output shape of this model

    :param forecastHorizon: forecast Horizon
    :param targetSeries train target series
    :param exogenousSeries train exogenous series
    :param seqLength train sequence length
    :param numExoVariables: number of exogenous variables
    :param targetEval: eval target series
    :param exoEval: eval exogenous series (can be None)
    """

    model = ExtremeTime2(
        forecastHorizon=forecastHorizon,
        memorySize=5,
        windowSize=5,
        embeddingSize=5,
        contextSize=5,
        numExoVariables=numExoVariables
    )

    model.train(targetSeries, seqLength, exogenousSeries, returnLosses=False)

    _, evalOut = model.evaluate(targetEval, exoEval, returnPred=True)
    assert evalOut.shape == (targetEval.shape[0] - forecastHorizon,)
