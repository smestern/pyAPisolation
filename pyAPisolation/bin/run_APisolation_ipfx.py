
print("Loading...")
import sys
sys.path.append('..')
import numpy as np
from numpy import genfromtxt
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
from scipy.optimize import curve_fit
from ipfx import feature_extractor
from ipfx import subthresh_features as subt
import pyabf
import logging
import scipy.ndimage as ndimage

from abf_utils import *
from abf_subthres import *
from abf_ipfx_dataframes import *





print("Load finished")
logging.basicConfig(level=logging.DEBUG)
root = tk.Tk()
root.withdraw()
files = filedialog.askdirectory(
                                   title='Select Dir'
                                   )
root_fold = files

##Declare our options at default


print('loading protocols...')
protocol_n = load_protocols(files)

filter = input("Allen's Gaussian Filter (recommended to be set to 0): ")
try: 
    filter = int(filter)
except:
    filter = 0

savfilter = input("Savitzky-Golay Filter (recommended to be set in 0): ")
try: 
    savfilter = int(savfilter)
except:
    savfilter = 0

tag = input("tag to apply output to files: ")
try: 
    tag = str(tag)
except:
    tag = ""
plot_sweeps = input("Enter the sweep Numbers to plot [seperated by a comma] (0 to plot all sweeps, -1 to plot no sweeps): ")
try:
    plot_sweeps = np.fromstring(plot_sweeps, dtype=int, sep=',')
    if plot_sweeps.shape[0] < 1:
        plot_sweeps = np.array([-1])
except:
    plot_sweeps = -1

print("protocols")
for i, x in enumerate(protocol_n):
    print(str(i) + '. '+ str(x))
proto = input("enter Protocol to analyze: ")
try: 
    proto = int(proto)
except:
    proto = 0

protocol_name = protocol_n[proto]
dv_cut = input("Enter the threshold cut off for the derivative (Allen defaults 20mv/s): ")
try: 
    dv_cut = int(dv_cut)
except:
    dv_cut = 20
tp_cut = input("Enter the threshold cut off for max threshold-to-peak time (Allen defaults 5ms)[in ms]: ")
try: 
    tp_cut = (np.float64(tp_cut)/1000)
except:
    tp_cut = 0.005

min_cut = input("Enter the minimum cut off for threshold-to-peak voltage (Allen defaults 2mV)[in mV]: ")
try: 
    min_cut = np.float64(min_cut)
except:
    min_cut = 2


min_peak = input("Enter the mininum cut off for peak voltage (Allen defaults -30mV)[in mV]: ")
try: 
    min_peak = np.float64(min_peak)
except:
    min_peak = -30

percent = input("Enter the percent of max DvDt used to calculate refined threshold (does not effect spike detection)(Allen defaults 5%)[in %]: ")
try: 
    percent = percent /100
except:
    percent = 5/100

lowerlim = input("Enter the time to start looking for spikes [in s] (enter 0 to start search at beginning): ")
upperlim = input("Enter the time to stop looking for spikes [in s] (enter 0 to search the full sweep): ")

try: 
    lowerlim = float(lowerlim)
    upperlim = float(upperlim)
except:
    upperlim = 0
    lowerlim = 0


print(f"Running analysis with, dVdt thresh: {dv_cut}mV/s, thresh to peak max: {tp_cut}s, thresh to peak min height: {min_cut}mV, and min peak voltage: {min_peak}mV")
param_dict = {'filter': filter, 'dv_cutoff':dv_cut, 'start': lowerlim, 'end': upperlim, 'max_interval': tp_cut, 'min_height': min_cut, 'min_peak': min_peak, 'thresh_frac': percent}
debugplot = 0
running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width']
dfs = pd.DataFrame()
df_spike_count = pd.DataFrame()
df_running_avg_count = pd.DataFrame()
for root,dir,fileList in os.walk(files):
 for filename in fileList:
    if filename.endswith(".abf"):
            file_path = os.path.join(root,filename)
        #try:
            abf = pyabf.ABF(file_path)
        
            if abf.sweepLabelY != 'Clamp Current (pA)' and protocol_name in abf.protocol:
                print(filename + ' import')
                temp_spike_df, df, temp_running_bin = analyze_abf(abf, sweeplist=None, plot=plot_sweeps, param_dict=param_dict)
                df_running_avg_count = df_running_avg_count.append(temp_running_bin)
                df_spike_count = df_spike_count.append(temp_spike_df, sort=True)
                dfs = dfs.append(df, sort=True)
            else:
                print('Not correct protocol: ' + abf.protocol)
        #except:
          #  print('Issue Processing ' + filename)

try:
 
    ids = dfs['__file_name'].unique()
    print(f"Ran analysis with, dVdt thresh: {dv_cut}mV/s, thresh to peak max: {tp_cut}s, thresh to peak min height: {min_cut}mV, and min peak voltage: {min_peak}mV")

    settings_col = ['dvdt Threshold', 'threshold to peak max time','threshold to peak min height', 'min peak voltage', 'allen filter', 'sav filter', 'protocol_name']
    setdata = [dv_cut, tp_cut, min_cut, min_peak, filter, savfilter, protocol_name]
    settings_df =  pd.DataFrame(data=[setdata], columns=settings_col, index=[0])
    settings_df.to_csv(root_fold + '/analysis_settings_' + tag + '.csv')
    tempframe = dfs.groupby('__file_name').mean().reset_index()
    tempframe.to_csv(root_fold + '/allAVG_' + tag + '.csv')

 
    tempframe = dfs.drop_duplicates(subset='__file_name')
    tempframe.to_csv(root_fold + '/allRheo_' + tag + '.csv')

 
    df_spike_count.to_csv(root_fold + '/spike_count_' + tag + '.csv')
    dfs.to_csv(root_fold + '/allfeatures_' + tag + '.csv')
    with pd.ExcelWriter(root_fold + '/running_avg_' + tag + '.xlsx') as runf:
        cols = df_running_avg_count.columns.values
        df_ind = df_running_avg_count.loc[:,cols[[-1,-2,-3]]]
        index = pd.MultiIndex.from_frame(df_ind)
        for p in running_lab:
            temp_ind = [p in col for col in cols]
            temp_df = df_running_avg_count.set_index(index).loc[:,temp_ind]
            temp_df.to_excel(runf, sheet_name=p)
except: 
    print('error saving')

print("==== SUCCESS ====")
input('Press ENTER to exit')