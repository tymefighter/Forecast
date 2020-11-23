import numpy as np

from ts.data.generate.univariate.nonexo import *
from ts.plot.static_plot import Plot


def main():
    n = 50

    dataGen = StandardGenerator()
    Plot.plotDataCols(dataGen.generate(n))

    dataGen = PeriodicGenerator()
    Plot.plotDataCols(dataGen.generate(n))

    dataGen = PolynomialGenerator()
    Plot.plotDataCols(dataGen.generate(n))

    dataGen = DifficultGenerator()
    Plot.plotDataCols(dataGen.generate(n))

    data = np.stack([
        dataGen.generate(n),
        dataGen.generate(n),
        dataGen.generate(n)
    ], axis=1)

    Plot.plotDataCols(data)


if __name__ == '__main__':
    main()
