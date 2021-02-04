import pytest
import os
import datetime
import pandas as pd
from ts.utility import DatasetUtility

testDataDir = './utility/testdata'


@pytest.mark.parametrize('filename, dateFormat', [
    ('data0.csv', '%Y-%m-%d'),
    ('data1.csv', '%Y-%m-%d'),
    ('data2.csv', '%d-%m-%Y'),
    ('data3.csv', '%m-%d-%Y')
])
def test_sortByDate(filename, dateFormat):
    filepath = os.path.join(testDataDir, filename)
    df = pd.read_csv(filepath)

    # Function we want to test
    newDf = DatasetUtility.sortByDate(df, dateFormat=dateFormat)

    # Getting the name of date column
    dateColumnName = next(filter(
        lambda columnName: columnName.lower() == 'date',
        list(df.columns)
    ))

    # Convert date string to datetime
    df[dateColumnName] = df[dateColumnName].map(
        lambda dateString: datetime.datetime.strptime(dateString, dateFormat)
    )

    # Dataframe date column must have dates in increasing order
    assert newDf[dateColumnName].is_monotonic_increasing

    # Index of dataframe must increase
    assert newDf.index.is_monotonic_increasing

    # Both dataframes after reordering of rows must be equal
    assert newDf.shape == df.shape

    for _, newDfRow in newDf.iterrows():
        oldDfRow = df\
            .loc[df[dateColumnName] == newDfRow[dateColumnName]]\
            .iloc[0]

        assert newDfRow.equals(oldDfRow)
