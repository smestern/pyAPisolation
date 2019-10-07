

import numpy as np
from numpy import genfromtxt
from abfderivative import *
from nuactionpotential import *
import pyabf
from pyabf.tools import *
import tkinter as tk
from tkinter import filedialog
import os
import pandas

root = tk.Tk()
root.withdraw()
files = filedialog.askopenfilenames(filetypes=(('ABF Files', '*.abf'),
                                   ('All files', '*.*')),
                                   title='Select Input File'
                                   )
fileList = root.tk.splitlist(files)

i = 0
k = 0
for filename in fileList:
    if filename.endswith(".abf"):
        i += 1
        file_path = filename
        abf = pyabf.ABF(file_path)
        if abf.sweepLabelY != 'Clamp Current (pA)':
            print(filename + ' import')
            np.nan_to_num(abf.data, nan=-9999, copy=False)
            tag = file_path.split('/')
            tag = tag[(len(tag) - 1)]
            #fileno, void = tag.split('-')
            thresholdavg(abf,0)
            _, df, _ = apisolate(abf, 0, "", False, True, plot=8)
        else: 
            k += 1
            print('current wrong')
                     
print(i)
print(k)
plt.show()