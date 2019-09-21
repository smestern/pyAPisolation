

import numpy as np
from numpy import genfromtxt
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from abfderivative import *
from nwbactionpotential import *
from datetime import datetime
from dateutil.tz import tzlocal

from pynwb import *
from pynwb.icephys import *

directory = 'nwb2\\specimen_313860745/'


for filename in os.listdir(directory):
    if filename.endswith(".nwb"):
        file_path = directory + filename
        io = NWBHDF5IO(file_path, 'r+')
        nwbfile_in = io.read()
            

plt.show()