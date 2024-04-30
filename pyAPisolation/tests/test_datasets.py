import numpy as np
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



    


test_x_y_c()