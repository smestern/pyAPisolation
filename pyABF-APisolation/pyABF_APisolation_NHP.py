

import numpy as np
from numpy import genfromtxt
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from abfderivative import *
from nuactionpotential import *
import pyabf
from pyabf.tools import *
from matplotlib import cm
import tkinter as tk
from tkinter import filedialog
import os
import gc


directory = 'NHP\\M03 (2018_11_15)'

for root, dirs, files in os.walk(directory):
   for filename in files:
    if filename.endswith(".abf"):
        file_path = root + '\\' + filename
        abf = pyabf.ABF(file_path)
        if abf.sweepLabelY != 'Clamp Current (pA)':
            print(filename + ' import')
            np.nan_to_num(abf.data, nan=-9999, copy=False)
            tag = file_path.split('/')
            tag = tag[(len(tag) - 1)]
            fileno, void = tag.split('.')
            tag = fileno[-17:]
            tag = file_path.split('\\')
            tag = fileno[-17:]
            thresholdavg(abf,0)

            apisolate(abf, 0, tag, True, True, plot=1)

            gc.collect()


plt.show()

            
            

