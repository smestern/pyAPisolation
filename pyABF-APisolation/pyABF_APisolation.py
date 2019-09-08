

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

            print(pyabf.tools.ap.ap_points_currentSweep(abf))
            print(filename + ' import')
            tag = file_path.split('/')
            thresholdavg(abf,0)
            #appreprocess3(abf, tag[(len(tag) - 1)], False, True)
            

plt.show()