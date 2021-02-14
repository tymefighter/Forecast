import pandas as pd


class IndianSubdivisionRainfall:

    @staticmethod
    def loadData(filepath):
        """
        Loads Indian Subdivision Rainfall Data as a Pandas Dataframe.
        The file is located in the 'datasets/' directory of this repository
        with name 'IndianSubdivisionRainfall.csv'

        :param filepath: path where the data file is located
        :return: complete dataframe of the loaded dataset
        """

        return pd.read_csv(filepath)
