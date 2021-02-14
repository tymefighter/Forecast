import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

SELF_DIR = 'nifty50-stock-market-data'


class NiftyFiftyStock:

    def __init__(self, dataPath, quiet=True):
        """
        Downloads the Dataset if not already present at the
        path specified. If the specified data path is not a
        directory (i.e. does not exist yet) or is empty, then
        the dataset is downloaded and then loaded, else it is
        just loaded from the directory path specified

        :param dataPath: Path of the directory where to download
        the dataset or from where to load the dataset
        :param quiet: If False, then downloading outputs from
        Kaggle-API would be displayed
        """

        self.dirPath = os.path.join(dataPath, SELF_DIR)

        if not os.path.isdir(self.dirPath):
            api = KaggleApi()
            api.authenticate()
            api.dataset_download_files(
                dataset='rohanrao/nifty50-stock-market-data',
                path=dataPath,
                quiet=quiet,
                unzip=True
            )

    def getFileNames(self):
        """ Get the names of all the files in this dataset"""

        return os.listdir(self.dirPath)

    def loadFile(self, filename):
        """
        Load the file with the given name (not filepath) from the
        loaded dataset.

        :param filename: Name of the file in the dataset which is
        to be loaded as a pandas dataframe. Note that this argument
        is the file name and not the file path.
        :return: Dataframe constructed from the file in the dataset
        whose filename is specified.
        """

        return pd.read_csv(os.path.join(self.dirPath, filename))
