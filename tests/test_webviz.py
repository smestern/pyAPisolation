from pyAPisolation.webViz.tsDatabase import tsDatabaseViewer
from pyAPisolation.webViz.ephysDatabase import ephysDatabaseViewer
import pandas as pd
from joblib import load
import os


def test_tsDatabase():
    # Test the tsDatabase class
    data_file = load(f'{os.path.dirname(__file__)}/test_data/known_good_df.joblib')
    db = tsDatabaseViewer(data_file, file_index='filename', folder_path='foldername')
    assert isinstance(db.database, pd.DataFrame)


def test_ephysDatabase():
    # Test the ephysDatabase class
    data_file = load(f'{os.path.dirname(__file__)}/test_data/known_good_df.joblib')
    db = ephysDatabaseViewer(data_file, file_index='filename', folder_path='foldername')
    assert isinstance(db.database, pd.DataFrame)

if __name__ == "__main__":
    test_tsDatabase()
    print("tsDatabase tests passed successfully.")