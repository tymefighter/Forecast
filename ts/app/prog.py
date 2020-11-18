import tensorflow as tf

from ts.model.univariate import LstmForecast
from ts.data.univariate.nonexo import StandardGenerator
from ts.log import GlobalLogger


def main():
    GlobalLogger.getLogger().setLevel(2)

    data = StandardGenerator('simple').generate(100)
    model = LstmForecast(1, 10, 'relu', 2)
    model.train([data], 5, verboseLevel=2, modelSavePath='/Users/ahmed/model')

    model = LstmForecast(modelLoadPath='/Users/ahmed/model')
    print(model.evaluate(data))


if __name__ == '__main__':
    main()
