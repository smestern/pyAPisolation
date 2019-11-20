
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pyabf
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import pyAPisolation as apis
from sklearn.ensemble import IsolationForest
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

featcon = input("save feature arrays all-in-one file? (y/n): ")
try: 
    feat = str(featcon)
except:
    feat = "n"
if feat == "n" or feat =="N":
    bfeatcon = False
else: 
    bfeatcon = True
    bfeat = False


if bfeatcon == True:
    featfile = input("save feature arrays all-in-one file averaging per input file? (y/n): ")
    try: 
        featfile = str(featfile)
    except:
        featfile = "n"
    if featfile == "n" or featfile =="N":
        featfile = False
    else: 
        featfile = True

    featrheo = input("save feature arrays all-in-one file rheobase only? (y/n): ")
    try: 
        featrheo = str(featrheo)
    except:
        featrheo = "n"
    if featrheo == "n" or featrheo =="N":
        featrheo = False
    else: 
        featrheo = True


    boutlier = input("Perform Outlier Elim? (y/n): ")
    try: 
        boutlier = str(boutlier)
    except:
        boutlier = "n"
    if boutlier == "n" or boutlier =="N":
        boutlier = False
    else: 
        boutlier = True
        

debugplot = input("return a plot of sample action potentials from the files (debug) (int): ")
try:
    debugplot = int(debugplot)
except:
    debugplot = 0

dfs = pd.DataFrame()

for filename in fileList:
    if filename.endswith(".abf"):
        file_path = filename
        abf = pyabf.ABF(file_path)
        abf.abfID
        if abf.sweepLabelY != 'Clamp Current (pA)':
            print(filename + ' import')
            np.nan_to_num(abf.data, nan=-9999, copy=False)
            _, df, _ = apis.nuactionpotential.apisolate(abf, filter, tag, braw, bfeat, plot=debugplot)
            df = df.assign(file_name=np.full(len(df.index),abf.abfID))
            
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df = df[cols]
            if bfeatcon == True:
               dfs = dfs.append(df)
        else:
            print('Not Current CLamp')
   
if boutlier == True:
    od = IsolationForest(contamination=0.01)
    d_out = dfs.iloc[:,2:].to_numpy()
    d_out = np.nan_to_num(d_out, False, 0.0)
    f_outliers = od.fit_predict(d_out)
    drop_o = np.nonzero(np.where(f_outliers==-1, 1, 0))[0]         

    outliers = dfs.iloc[drop_o].copy(deep=True)
    dfs = dfs.drop(dfs.index[drop_o], axis=0)
    outliers.to_csv('output/outliers_' + tag + '.csv')

if featfile == True:
    ids = dfs['file_name'].unique()
    tempframe = dfs.groupby('file_name').mean().reset_index()
    tempframe.to_csv('output/allAVG' + tag + '.csv')

if featrheo == True:
    tempframe = dfs.drop_duplicates(subset='file_name')
    tempframe.to_csv('output/allRheo' + tag + '.csv')

        




if bfeatcon == True:
    dfs.to_csv('output/allfeat' + tag + '.csv')
    



plt.show()