print("Loading...")
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy import stats
from ipfx import feature_extractor
from ipfx import subthresh_features as subt
from ipfx import feature_vectors as fv
from ipfx.sweep import Sweep
from sklearn.preprocessing import minmax_scale
import pyabf
import logging
import scipy.ndimage as ndimage
def loadABF(file_path, return_obj=False):
    '''
    Employs pyABF to generate numpy arrays of the ABF data. Optionally returns abf object.
    Same I/O as loadNWB
    '''
    abf = pyabf.ABF(file_path, cacheStimulusFiles=True)
    dataX = []
    dataY = []
    dataC = []
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        tempX = abf.sweepX
        tempY = abf.sweepY
        tempC = abf.sweepC
        dataX.append(tempX)
        dataY.append(tempY)
        dataC.append(tempC)
    npdataX = np.vstack(dataX)
    npdataY = np.vstack(dataY)
    npdataC = np.vstack(dataC)

    if return_obj == True:

        return npdataX, npdataY, npdataC, abf
    else:

        return npdataX, npdataY, npdataC

    ##Final return incase if statement fails somehow
    return npdataX, npdataY, npdataC

def find_peak_abs(ar):
    peak_idx = np.argmax(np.abs(ar))
    return ar[peak_idx], peak_idx


print("Load finished")

folder = 'H:\\Sam\\2104\\015_PERF_PATCH\\'
abf_files = glob.glob(folder + "*.abf")
stim = 'H:\\Sam\\test.abf'
_, stim_c, _ = loadABF(stim)
intercept = []
x_y_data = []
ids = []
df_full = pd.DataFrame()
for x in abf_files:
    abf = pyabf.ABF(x, loadData=False)
    if 'Puff' in abf.protocol:
        x, y, _, abf = loadABF(x, return_obj=True)
        baseline_low, baseline_high = int(abf.dataRate * 0.5), int(abf.dataRate * 1.1)
        search_low, search_high = int(abf.dataRate * 1.5), int(abf.dataRate * 4)
        c = stim_c[:x.shape[0], :] + -70
        holding = c[:, search_low]
        #lowpass 
        sos = signal.bessel(8, 10, fs=abf.dataRate, output='sos')
        y = signal.sosfilt(sos, np.hstack([y,y]), axis=1)[:, x.shape[1]:]#signal.savgol_filter(y, 101, 3, axis=1, mode='mirror')
        holding_uni, uni_idx = np.unique(holding, return_index=True)
        y_mean = np.vstack([np.nanmean(a, axis=0) for a in np.split(y, uni_idx[1:], axis=0)])


        sweepwise_baseline = np.nanmean(y_mean[:, baseline_low:baseline_high], axis=1)
        y_mean_baseline = []
        for i, f in enumerate(sweepwise_baseline):
            y_mean_baseline.append(y_mean[i, :] - f)
        y_mean_baseline = np.vstack(y_mean_baseline)
        peak_peak_idx = np.apply_along_axis(find_peak_abs, 1, y_mean_baseline[:, search_low:search_high])
        peak_peak_idx = np.vstack(peak_peak_idx)
        peak, peak_idx = peak_peak_idx[:, 0], peak_peak_idx[:,1] + search_low
        relative_peak = peak 
        x_y_data.append([holding_uni, relative_peak])

        lin_res = stats.linregress(holding_uni, relative_peak)
        #y = mx+b
        x_int = (0 - lin_res.intercept) / lin_res.slope
        intercept.append(x_int)
        ids.append(abf.abfID)

        plt.clf()
        plt.subplot(2,1,1)
        for i in np.arange(y_mean.shape[0]):
            plt.plot(x[0, :], y_mean_baseline[i,:], label=f"Holding {holding_uni[i]}")
            plt.scatter(x[0,int(peak_idx[i])], y_mean_baseline[i, int(peak_idx[i])])
            plt.axvline(1.5)
        plt.subplot(2,1,2)
        plt.scatter(holding_uni, relative_peak)
        lin_r = lin_res.slope * np.arange(-100, -20, step=10) + lin_res.intercept
        plt.plot(np.arange(-100, -20, step=10), lin_r)
        plt.axhline(0)
        plt.savefig(f"{abf.abfFilePath}.png")

        df_temp = pd.DataFrame(data=relative_peak.reshape(1, -1), columns=holding_uni, index=[abf.abfID])
        df_temp['x_intercept'] = x_int
        df_full = df_full.append(df_temp, sort=True)
        print(df_temp)
df_full.to_csv(folder+"puff_analysis.csv")

