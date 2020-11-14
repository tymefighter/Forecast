import numpy as np

from ts.data.univariate.nonexo import *
from ts.plot.static_plot import Plot


def main():
    n = 50

    dataGen = StandardGenerator()
    Plot.plotDataCols(np.expand_dims(dataGen.generate(n), axis=1))

    dataGen = PeriodicGenerator()
    Plot.plotDataCols(np.expand_dims(dataGen.generate(n), axis=1))

    dataGen = PolynomialGenerator()
    Plot.plotDataCols(np.expand_dims(dataGen.generate(n), axis=1))

    dataGen = DifficultGenerator()
    Plot.plotDataCols(np.expand_dims(dataGen.generate(n), axis=1))


if __name__ == '__main__':
    main()