import numpy as np
import tensorflow as tf

from ts.data.univariate.nonexo import ArmaGenerator
from ts.plot import Plot


def simpleData(n):
    obsCoef = np.array([0.5, -0.2, 0.2])
    noiseCoef = np.array([0.5, -0.2])

    noiseGenFunc = np.random.normal
    noiseGenParams = (0.0, 1.0)

    return ArmaGenerator(obsCoef, noiseCoef, noiseGenFunc, noiseGenParams) \
        .generate(n)


def prepareData(data, trainSize):

    train = data[:trainSize]
    test = data[trainSize:]

    n = train.shape[0] - 1
    x = np.expand_dims(train[:n], axis=[0, 2])
    y = np.expand_dims(train[1:], axis=[0, 2])

    m = test.shape[0] - 1
    xt = np.expand_dims(test[:m], axis=[0, 2])
    yt = np.expand_dims(test[1:], axis=[0, 2])

    return x, y, xt, yt


def train1(x, y, xt, yt, sz=100):

    model = tf.keras.Sequential([
        tf.keras.layers.GRU(10, input_shape=(None, 1), return_sequences=True),
        tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(1, activation='linear'))
    ])

    model.compile(tf.optimizers.Adam(0.3), 'mse')
    losses = []

    for i in range(30):
        print(i)
        start = 0
        while start < x.shape[0]:
            end = min(start + sz, x.shape[0])
            history = model.fit(x[start:end], y[start:end], batch_size=1, epochs=1, verbose=0)
            losses.extend(history.history['loss'])
            start = end

    Plot.plotLoss(losses)

    yPred = np.squeeze(model.predict(x, batch_size=1))
    Plot.plotPredTrue(yPred, np.squeeze(y))

    yPred = np.squeeze(model.predict(xt, batch_size=1))
    Plot.plotPredTrue(yPred, np.squeeze(yt))


def main():

    data = simpleData(5000)
    prepData = prepareData(data, 4500)
    train1(*prepData)


if __name__ == '__main__':
    main()