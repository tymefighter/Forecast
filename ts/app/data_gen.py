import numpy as np

from ts.data.univariate.nonexo import *
from ts.plot.static_plot import Plot


def main():
    n = 500

    data = StandardGenerator('extreme_short').generate(n)
    Plot.plotDataCols(data)

    data = StandardGenerator('extreme_long').generate(n)
    Plot.plotDataCols(data)

    data = StandardGenerator('long_term').generate(n)
    Plot.plotDataCols(data)


if __name__ == '__main__':
    main()