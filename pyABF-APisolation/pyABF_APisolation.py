

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