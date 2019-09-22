

import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from abfderivative import *
from nwbactionpotential import *
from datetime import datetime
from dateutil.tz import tzlocal
from pynwb import *
from pynwb.icephys import *
import ipfx.nwb_reader as nwb_reader
directory = 'nwb2\\specimen_313860745\\'


for filename in os.listdir(directory):
    if filename.endswith(".nwb"):
        file_path = directory + filename
        filedata = nwb_reader.create_nwb_reader(file_path)

            

plt.show()