# Time Series Forecasting Package

This is a `python` package containing Various Forecasting Algorithms,
Forecasting Datasets and, Plotting, Preprocessing and Utility Tools.

## Forecasting Algorithms

This package provides (or would provide) various algorithms which work
on data containing multivariate target time series, univariate target
time series - both with or without exogenous time series. The algorithms
are provided in the `model` subpackage of the time series package `ts`.

Currently, this package provides the following Forecasting Algorithms,

-   MLP (DNN) based multivariate forecasting algorithm which has support
    for exogenous time series. You may import this in your application by
    doing as follows

```
from ts.model.multivariate import DeepNN
```

-   RNN based univariate forecasting algorithm with support for multivariate 
    exogenous time series. This is a recurrent model which takes in any
    recurrent layer (and parameters) and stacks it required number of times.

```
from ts.model.univariate import RnnForecast
```

-   Simple RNN based univariate forecasting algorithm with support for multivariate 
    exogenous time series. This model is built by stacking multiple simple RNN layers.

```
from ts.model.univariate import SimpleRnnForecast
```

-   GRU based univariate forecasting algorithm with support for multivariate 
    exogenous time series. This model is built by stacking multiple GRU layers.

```
from ts.model.univariate import GruForecast
```

-   LSTM based univariate forecasting algorithm with support for multivariate 
    exogenous time series. This model is built by stacking multiple LSTM layers.

```
from ts.model.univariate import LstmForecast
```

-   The Extreme Time Model - which focuses on forecasting target series with
    extreme values - i.e. values which have a very large deviation from the
    time series trend. It supports univariate time series and multivariate 
    exogenous time series.

```
from ts.model.special import ExtremeTime
```

-   The Extreme Time Model 2 - It is another model focussed on forecasting
    time series with extreme values. It supports univariate time series and
    multivariate exogenous time series.

```
from ts.model.special import ExtremeTime2
```

## Forecasting Data

This package provides Data Generators as well as Datasets. The `data`
subpackage of the time series package `ts` provides two subpackages
named `generate` and `dataset`, the first one contains data generators
and the second one contains real world datasets.

Currently, we provide the following,

-   ARMA Generated data - generates only univariate target series
    without support for exogenous series.

```
from ts.model.univariate.nonexo import ArmaGenerator
```

-   Standard Generators - provides generators for simple data, long
    term dependency data and extreme valued data

```
from ts.model.univariate.nonexo import StandardGenerator
```

-   Periodic Generator - generates periodic data but with support for
    only univariate target series without exogenous series.

```
from ts.model.univariate.nonexo import PeriodicGenerator
```

-   Polynomial Generator - generates data which is a polynomial function
    of time. This allows one to generate data with a varying trend.

```
from ts.model.univariate.nonexo import PolynomialGenerator
```

-   Difficult Generator - generates difficult data, i.e. data which has
    a varying trend, periodicity (seasonality) and noise (ARMA). One
    can introduce extreme values into the data by providing the appropriate
    contructor arguments

```
from ts.model.univariate.nonexo import DifficultGenerator
```

## Plotting, Preprocessing and Utility Tools

This package contains plotting tools for plotting losses, plotting training
data and comparing prediction with true (using plots). The plotting tools
are available in the `plot` subpackage of the time series package `ts`.
It also provides utility tools for in the `utility` subpackage of `ts`.
Access to the global logger and local loggers is provided in the `log`
subpackage of `ts`.

## Package Structure

```
ts/
|__ data/
|    |__ dataset/
|         |__ AmazonStockPrice
|
|    |__ generate/
|         |__ univariate/
|              |__ nonexo/
|                  |__ ArmaGenerator
|                  |__ StandardGenerator
|                  |__ PeriodicGenerator
|                  |__ PolynomialGenerator
|                  |__ DifficultGenerator
| 
|__ model/
|    |__ univariate/
|         |__ RnnForecast
|         |__ SimpleRnnForecast
|         |__ GruForecast
|         |__ LstmForecast
|
|    |__ multivariate/
|         |__ DeepNN
|
|    |__ special/
|         |__ ExtremeTime
|         |__ ExtremeTime2
|
|__ plot/
|    |__ Plot
|
|__ log/
|    |__ ConsoleLogger
|    |__ FileLogger
|    |__ GlobalLogger
|
|__ utility/
|    |__ Utility
|    |__ ForecastDataSequence
|    |__ SaveCallback
```

## Repository Structure

This repository is structured as follows:

```
Forecast
|__ other/...
|
|__ notebooks/..
|
|__ ts/...
```

- `other`:  contains deprecated codes and codes that do not work.
    The contents of this directory would be removed as soon as
    they have been analyzed thoroughly.

- `notebooks`: contains notebooks containing experiments, tests and examples

- `ts`: The time series forecasting package

## Existing experiments, tests and example notebooks

The `notebooks` directory of this repository contains notebooks
containing experiments, tests and examples. To be able to run
these notebooks, we need to do as follows:

1. Go to your .bashrc file and add the following line:
```
export PYTHONPATH="$PYTHONPATH:<location_of_this_repository>"
```

where `<location_of_this_repository>` is the location of this repository
in your filesystem.


