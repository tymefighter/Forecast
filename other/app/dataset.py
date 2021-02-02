from ts.data.dataset.univariate import AmazonStockPrice


def main():

    targetSeries, exogenousSeries = AmazonStockPrice.loadData(
        '/Users/ahmed/Downloads/Datasets/amazonStockPrice',
        'Open'
    )
    print(targetSeries.shape)
    print(targetSeries.dtype)
    print(targetSeries[:10])

    print(exogenousSeries.shape)
    print(exogenousSeries.dtype)
    print(exogenousSeries[:10])


if __name__ == '__main__':
    main()
