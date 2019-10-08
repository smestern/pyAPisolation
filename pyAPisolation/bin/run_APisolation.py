
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pyabf
import tkinter as tk
from tkinter import filedialog
import os
import pandas
import pyAPisolation as apis
root = tk.Tk()
root.withdraw()
files = filedialog.askopenfilenames(filetypes=(('ABF Files', '*.abf'),
                                   ('All files', '*.*')),
                                   title='Select Input File'
                                   )
fileList = root.tk.splitlist(files)

for filename in fileList:
    if filename.endswith(".abf"):
        file_path = filename
        abf = pyabf.ABF(file_path)
        if abf.sweepLabelY != 'Clamp Current (pA)':
            print(filename + ' import')
            np.nan_to_num(abf.data, nan=-9999, copy=False)
            tag = file_path.split('/')
            tag = tag[(len(tag) - 1)]
            #fileno, void = tag.split('-')
            apis.nuactionpotential.thresholdavg(abf,0)
            _, df, _ = apis.nuactionpotential.apisolate(abf, 0, "", False, True, plot=8)
        else:
            print('Not Current CLamp')
                     

plt.show()