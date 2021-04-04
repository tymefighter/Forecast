import pandas as pd
from ts.utility import DatasetUtility


class AppleStock:

    @staticmethod
    def loadData(filepath):
        """
        Loads Apple Stock Data as a Pandas Dataframe.
        The file is located in the 'datasets/' directory of this repository
        with name 'apple-stock.csv'

        :param filepath: path where the data file is located
        :return: complete dataframe of the loaded dataset
        """

        return DatasetUtility.sortByDate(pd.read_csv(filepath, skiprows=14))
