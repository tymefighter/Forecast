import tensorflow as tf

from ts.model.univariate.multiseq.deep import RnnForecast
from ts.data.univariate.nonexo import StandardGenerator
from ts.log import GlobalLogger


def main():
    GlobalLogger.getLogger().setLevel(2)

    data = StandardGenerator('simple').generate(100)
    model = RnnForecast(1, tf.keras.layers.SimpleRNN, 10, 1, 0)
    model.train([data])


if __name__ == '__main__':
    main()