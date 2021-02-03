import pandas as pd
import datetime


class DatasetUtility:

    @staticmethod
    def sortByDate(df: pd.DataFrame, dateFormat='%Y-%m-%d'):
        """
        Sorts the dataset rows based on the date if not already sorted
        after converting the date from string to a datetime object.

        :param df: The dataset in the form of a dataframe
        :param dateFormat: Format of the date present in the dataset
        :return:
        """

        dateColumnName = next(filter(
            lambda columnName: columnName.lower() == 'date',
            list(df.columns)
        ))

        df[dateColumnName] = df[dateColumnName].map(
            lambda dateString: datetime.datetime.strptime(dateString, dateFormat)
        )

        if not df[dateColumnName].is_monotonic_increasing:
            df = df \
                .sort_values(by=[dateColumnName]) \
                .reset_index(drop=True)

        return df
