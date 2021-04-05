import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from ts.utility import DatasetUtility


class DelhiClimate:

    @staticmethod
    def loadData(dataPath):
        """
        Loads the Delhi Climate Dataset as a dataframe. If the
        dataset is not present at the path, then it downloads the
        dataset. The returned dataset is sorted by date

        :param dataPath: filepath of where to download the dataset (or where
        the dataset is located if the dataset is already downloaded)
        :return: complete dataframe of the loaded dataset
        """

        if not os.path.isdir(dataPath):
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                dataset='sumanthvrao/daily-climate-time-series-data',
                path=dataPath,
                quiet=True,
                unzip=True
            )

        filepath1 = os.path.join(dataPath, 'DailyDelhiClimateTrain.csv')
        df1 = DatasetUtility.sortByDate(pd.read_csv(filepath1, header='infer'))

        filepath2 = os.path.join(dataPath, 'DailyDelhiClimateTest.csv')
        df2 = DatasetUtility.sortByDate(pd.read_csv(filepath2, header='infer'))

        return pd.concat([df1, df2], axis=0, ignore_index=True)
