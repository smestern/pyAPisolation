

import numpy as np
from numpy import genfromtxt
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from abfderivative import *
from nwbactionpotential import *
from datetime import datetime
from dateutil.tz import tzlocal

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.stimulus_info as si

from pynwb import NWBFile, NWBHDF5IO, TimeSeries
from pynwb.ophys import OpticalChannel, DfOverF, ImageSegmentation
from pynwb.image import ImageSeries, IndexSeries
from pynwb.device import Device

# Settings:
ophys_experiment_id = 562095852
save_file_name = 'brain_observatory.nwb'


directory = 'Processed/'


for filename in os.listdir(directory):
    if filename.endswith(".abf"):
        file_path = directory + filename
        abf = pyabf.ABF(file_path)
        if abf.sweepLabelY != 'Clamp Current (pA)':
            print(filename + ' import')
            np.nan_to_num(abf.data, nan=-9999, copy=False)
            tag = file_path.split('/')
            tag = tag[(len(tag) - 1)]
            #fileno, void = tag.split('-')
            thresholdavg(abf,0)
            apisolate(abf, 0, tag, False, True, plot=1)
            

plt.show()