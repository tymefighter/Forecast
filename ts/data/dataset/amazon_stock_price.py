import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from ts.utility import DatasetUtility


class AmazonStockPrice:

    @staticmethod
    def loadData(dataPath):
        """
        Loads the Amazon Stock Price Dataset as a dataframe. If the
        dataset is not present at the path, then it downloads the
        dataset. The returned dataset is sorted by date

        :param dataPath: filepath of where to download the dataset (or where
        the dataset is located if the dataset is already downloaded)
        :return: complete dataframe of the loaded dataset
        """

        filename = 'Amazon.csv'
        filePath = os.path.join(dataPath, filename)

        if not os.path.isfile(filePath):
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_file(
                dataset='salmanfaroz/amazon-stock-price-1997-to-2020',
                file_name=filename,
                path=dataPath
            )

        return DatasetUtility.sortByDate(pd.read_csv(filePath, header='infer'))
