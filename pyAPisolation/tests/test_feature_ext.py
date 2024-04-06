import numpy as np
from pyAPisolation.loadNWB import loadFile
from pyAPisolation.dataset import cellData

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
    


test_x_y_c()