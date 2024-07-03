from pyAPisolation.webViz.tsDatabase import tsDatabase
from pyAPisolation.webViz.ephysDatabase import ephysDatabase
import pandas as pd
from joblib import load
import os


def test_tsDatabase():
    # Test the tsDatabase class
    data_file = load(f'{os.path.dirname(__file__)}/test_data/known_good_df.joblib')
    db = tsDatabase(data_file, file_index='filename', folder_path='foldername')
    assert isinstance(db.database, pd.DataFrame)


def test_ephysDatabase():
    # Test the ephysDatabase class
    data_file = load(f'{os.path.dirname(__file__)}/test_data/known_good_df.joblib')
    db = ephysDatabase(data_file, file_index='filename', folder_path='foldername')
    assert isinstance(db.database, pd.DataFrame)

if __name__ == "__main__":
    test_tsDatabase()
    print("tsDatabase tests passed successfully.")