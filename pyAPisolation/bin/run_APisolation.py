
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

##Declare our options at default
filter = input("Filter (recommended to be set to 0): ")
braw = False
bfeat = True
try: 
    filter = int(filter)
except:
    filter = 0
tag = input("tag to apply output to files: ")
try: 
    tag = str(tag)
except:
    tag = ""
raw = input("save raw action potential traces? (y/n): ")
try: 
    raw = str(raw)
except:
    raw = "n"
if raw == "y" or raw =="Y":
    braw = True

feat = input("save feature arrays for each file? (y/n): ")
try: 
    feat = str(feat)
except:
    feat = "n"
if feat == "n" or feat =="N":
    bfeat = False
else: 
    bfeat = True

debugplot = input("return a plot of sample action potentials from the files (debug) (int): ")
try:
    debugplot = int(debugplot)
except:
    debugplot = 0



for filename in fileList:
    if filename.endswith(".abf"):
        file_path = filename
        abf = pyabf.ABF(file_path)
        if abf.sweepLabelY != 'Clamp Current (pA)':
            print(filename + ' import')
            np.nan_to_num(abf.data, nan=-9999, copy=False)
            apis.nuactionpotential.thresholdavg(abf,0)
            _, df, _ = apis.nuactionpotential.apisolate(abf, filter, tag, braw, bfeat, plot=debugplot)
        else:
            print('Not Current CLamp')
                     

plt.show()