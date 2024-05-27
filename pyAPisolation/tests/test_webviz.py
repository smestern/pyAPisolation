from pyAPisolation.web_viz.tsdatabase import tsDatabase
import pandas as pd
from joblib import load
import os


def test_tsDatabase():
    # Test the tsDatabase class
    data_file = load(f'{os.path.dirname(__file__)}/test_data/known_good_df.joblib')
    db = tsDatabase(data_file, file_index='filename', folder_path='foldername')
    assert isinstance(db.database, pd.DataFrame)
    assert db.database.shape == (10, 10)


if __name__ == "__main__":
    test_tsDatabase()
    print("tsDatabase tests passed successfully.")