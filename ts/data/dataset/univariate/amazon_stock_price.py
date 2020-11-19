import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


class AmazonStockPrice:

    @staticmethod
    def loadData(dataPath, targetVariable):
        filePath = dataPath + '/Amazon.csv'

        if not os.path.isfile(dataPath + dataPath):
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                'salmanfaroz/amazon-stock-price-1997-to-2020',
                dataPath,
                unzip=True
            )

        dataFrame = pd\
            .read_csv(filePath, header='infer')\
            .drop(columns='Date')

        assert (targetVariable in dataFrame.columns)

        exogenousSeries = dataFrame.iloc[:, dataFrame.columns != targetVariable].to_numpy()
        targetSeries = dataFrame[targetVariable].to_numpy()

        return targetSeries, exogenousSeries
