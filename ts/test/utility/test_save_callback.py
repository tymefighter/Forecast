import pytest
import pickle
import os
import numpy as np
import tensorflow as tf
from ts.utility import SaveCallback

FILE_PATH = 'utility/scratch/model'


class TestModel:
    """ Test Model - encapsulates tensorflow model for testing purposes """

    def __init__(self, model):
        """
        Takes as input a tensorflow model and saves it as a member

        :param model: a tensorflow model
        """

        self.model = model

    def save(self, modelSavePath):
        """
        Saves the model weights at the specified path

        :param modelSavePath: Path where to save the model weights
        """

        with open(modelSavePath, 'wb') as fl:
            pickle.dump(self.model.get_weights(), fl)


def checkWeightsEqual(weights1, weights2):
    """
    Checks if two tensorflow model weights are equal

    :param weights1: first list of weights
    :param weights2: second list of weights
    :return: True if both weights are equal, else false
    """

    if len(weights1) != len(weights2):
        return False

    for i in range(len(weights1)):
        if not np.array_equal(weights1[i], weights2[i]):
            return False

    return True


# Test Case 0
def getCase0():
    """ MLP Test Case """

    # Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10),
        tf.keras.layers.Dense(2)
    ])

    model.compile(loss='mse')

    # data
    X = np.random.uniform(-1, 1, size=(100, 10))
    Y = np.random.uniform(-1, 1, size=(100, 2))

    return TestModel(model), X, Y


# Test Case 1
def getCase1():
    """ RNN Test Case """

    # Model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(10, return_sequences=True),
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(2)
        )
    ])

    model.compile(loss=tf.keras.losses.MeanSquaredError())

    # data
    X = np.random.uniform(-1, 1, size=(1, 100, 10))
    Y = np.random.uniform(-1, 1, size=(1, 100, 2))

    return TestModel(model), X, Y


@pytest.mark.parametrize('genFunc', [
    getCase0, getCase1
], ids=['mlp-0', 'rnn-0'])
def test_SaveCallback(genFunc):

    # Get model and data
    testModel, X, Y = genFunc()

    # Train model on data, save model at each epoch
    testModel.model.fit(
        X, Y,
        callbacks=[SaveCallback(testModel, FILE_PATH)],
        epochs=1,
        verbose=0
    )

    # Get the saved weights
    with open(FILE_PATH, 'rb') as fl:
        savedWeights = pickle.load(fl)

    # Delete the file where model weights were saved
    os.remove(FILE_PATH)

    # The saved weights and current weights must match
    assert checkWeightsEqual(
        testModel.model.get_weights(),
        savedWeights
    )
