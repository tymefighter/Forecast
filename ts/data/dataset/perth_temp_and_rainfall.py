import os
import pandas as pd
import datetime
from kaggle.api.kaggle_api_extended import KaggleApi


class PerthTempAndRainfall:

    @staticmethod
    def loadData(dataPath):
        """
        Loads the Perth Temperature and Rainfall Dataset as a dataframe.
        If the dataset is not present at the path, then it downloads the
        dataset. The returned dataset is sorted by date

        :param dataPath: filepath of where to download the dataset (or where
        the dataset is located if the dataset is already downloaded)
        :return: complete dataframe of the loaded dataset
        """

        filename = 'PerthTemperatures.csv'
        filePath = os.path.join(dataPath, filename)

        if not os.path.isfile(filePath):
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_file(
                dataset='taranmarley/perth-temperatures-and-rainfall',
                file_name=filename,
                path=dataPath
            )

        perthTempAndRainfall = pd.read_csv(filePath, header='infer')
        perthTempAndRainfall['Date'] = perthTempAndRainfall.apply(
            lambda row: None if row.isnull().to_numpy().any() else datetime.date(
                int(row['Year']), int(row['Month']), int(row['Day'])
            ),
            axis=1
        )

        return perthTempAndRainfall \
            .sort_values(by=['Date']) \
            .reset_index(drop=True)
