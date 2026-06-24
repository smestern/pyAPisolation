import numpy as np
import pandas as pd
from gigaseal.dataset import cellData
from gigaseal.database import tsDatabase
import os

def test_x_y_c():
    #generate some fake data
    x = np.random.rand(10, 1000)
    y = np.random.rand(10, 1000)
    c = np.random.rand(10, 1000)

    #pass the data to the dataset object
    data = cellData(dataX=x, dataY=y, dataC=c)

    #check that the data is stored correctly
    assert np.all(data.dataX == x)

    #check that we generated a name for the data
    assert data.name is not None

    #check the other attributes are stored correctly
    assert data.sweepNumber == 0

    assert data.sweep == 0

    assert data.sweepList == list(range(10))

    data.setSweep(1)

    assert data.sweep == 1

    assert np.all(data.sweepX == x[1])


    #try it again passing a name
    data = cellData(dataX=x, dataY=y, dataC=c, name='test')

    #check that the data is stored correctly
    assert np.all(data.dataX == x)
    
    #check that we generated a name for the data
    assert data.name == 'test'

    assert data.sweepNumber == 0

    print('All tests passed')


def test_database():
    # Build a representative DataFrame inline so there is no dependency on a
    # serialised reference file (which can become incompatible with newer
    # pandas / joblib versions).
    df = pd.DataFrame({
        'filename': ['cell_001', 'cell_002', 'cell_003'],
        'IC1': ['/data/cell_001_IC1.abf', '/data/cell_002_IC1.abf', '/data/cell_003_IC1.abf'],
        'drug': ['ctrl', 'drug', 'ctrl'],
    })

    db = tsDatabase()
    success = db.from_dataframe(df, cell_id_col='filename',
                                 filename_cols=['IC1'],
                                 metadata_cols=['drug'])
    assert success, "from_dataframe should return True on success"
    print(db.cellindex.head())

    # Cells should be indexed by filename
    assert 'cell_001' in db.cellindex.index.values
    assert 'IC1' in db.cellindex.columns

    #check that we can add a new entry
    db.addEntry('test')
    assert 'test' in db.cellindex.index.values

    #try adding a protocol
    db.addProtocol('test', 'protocol', path='test_path')
    assert db.cellindex.loc['test', 'protocol'] == 'test_path'


if __name__ == "__main__":
    test_database()
    test_x_y_c()