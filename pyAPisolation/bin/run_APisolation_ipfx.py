
print("Loading...")
import sys
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
print("Load finished")
logging.basicConfig(level=logging.DEBUG)
root = tk.Tk()
root.withdraw()
files = filedialog.askdirectory(
                                   title='Select dir File'
                                   )
root_fold = files

##Declare our options at default

def exp_grow(t, a, b, alpha):
    return a - b * np.exp(-alpha * t)
def exp_decay_2p(t, a, b1, alphaFast, b2, alphaSlow):
    return a + b1*np.exp(-alphaFast*t) + b2*np.exp(-alphaSlow*t)
def exp_decay_1p(t, a, b1, alphaFast):
    return a + b1*np.exp(-alphaFast*t)
def exp_growth_factor(dataT,dataV,dataI, end_index=300):
    try:
        
        diff_I = np.diff(dataI)
        upwardinfl = np.argmax(diff_I)
        
        upperC = np.amax(dataV[upwardinfl:end_index])
        t1 = dataT[upwardinfl:end_index] - dataT[upwardinfl]
        curve = curve_fit(exp_grow, t1, dataV[upwardinfl:end_index], maxfev=50000, bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))[0]
        tau = curve[2]
        return 1/tau
    except:
        return np.nan



def exp_decay_factor(dataT,dataV,dataI, end_index=3000, abf_id='abf'):
     try:
        
        diff_I = np.diff(dataI)
        downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
        end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl)/2)
        
        upperC = np.amax(dataV[downwardinfl:end_index])
        lowerC = np.amin(dataV[downwardinfl:end_index])
        diff = np.abs(upperC - lowerC)
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        SpanFast=(upperC-lowerC)*1*.01
        curve, pcov_2p = curve_fit(exp_decay_2p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=([-np.inf,  0, 100,  0, 0], [np.inf, np.inf, 500, np.inf, np.inf]))
        curve2, pcov_1p = curve_fit(exp_decay_1p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=(-np.inf, np.inf))
        residuals_2p = dataV[downwardinfl:end_index]- exp_decay_2p(t1, *curve)
        residuals_1p = dataV[downwardinfl:end_index]- exp_decay_1p(t1, *curve2)
        ss_res_2p = np.sum(residuals_2p**2)
        ss_res_1p = np.sum(residuals_1p**2)
        ss_tot = np.sum((dataV[downwardinfl:end_index]-np.mean(dataV[downwardinfl:end_index]))**2)
        r_squared_2p = 1 - (ss_res_2p / ss_tot)
        r_squared_1p = 1 - (ss_res_1p / ss_tot)
        
        tau1 = 1/curve[2]
        tau2 = 1/curve[4]
        fast = np.min([tau1, tau2])
        slow = np.max([tau1, tau2])
        return tau1, tau2, curve, r_squared_2p, r_squared_1p
     except:
        return np.nan, np.nan, np.array([np.nan,np.nan,np.nan,np.nan,np.nan]), np.nan, np.nan



print('loading protocols...')
protocol = []
for root,dir,fileList in os.walk(files):
 for filename in fileList:
    if filename.endswith(".abf"):
        try:
            file_path = os.path.join(root,filename)
            abf = pyabf.ABF(file_path, loadData=False)
            protocol = np.hstack((protocol, abf.protocol))
        except:
            print('error processing file ' + file_path)
protocol_n = np.unique(protocol)
filter = input("Allen's Gaussian Filter (recommended to be set to 0): ")
braw = False
bfeat = True
try: 
    filter = int(filter)
except:
    filter = 0

savfilter = input("Savitzky-Golay Filter (recommended to be set in 0): ")
braw = False
bfeat = True
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


bfeatcon = True
bfeat = False
print(f"Running analysis with, dVdt thresh: {dv_cut}mV/s, thresh to peak max: {tp_cut}s, thresh to peak min height: {min_cut}mV, and min peak voltage: {min_peak}mV")

if bfeatcon == True:
    featfile = "y"
    try: 
        featfile = str(featfile)
    except:
        featfile = "n"
    if featfile == "n" or featfile =="N":
        featfile = False
    else: 
        featfile = True

    featrheo = "y"
    try: 
        featrheo = str(featrheo)
    except:
        featrheo = "n"
    if featrheo == "n" or featrheo =="N":
        featrheo = False
    else: 
        featrheo = True
        
def plotabf(abf, spiketimes, lowerlim, upperlim, sweep_plots):
   try:
    if sweep_plots[0] == -1:
        pass
    else:
        plt.figure(num=2, figsize=(16,6))
        plt.clf()
        cm = plt.get_cmap("Set1") #Changes colour based on sweep number
        if sweep_plots[0] == 0:
            sweepList = abf.sweepList
        else:
            sweepList = sweep_plots - 1
        colors = [cm(x/np.asarray(sweepList).shape[0]) for x,_ in enumerate(sweepList)]
        
        plt.autoscale(True)
        plt.grid(alpha=0)

        plt.xlabel(abf.sweepLabelX)
        plt.ylabel(abf.sweepLabelY)
        plt.title(abf.abfID)

        for c, sweepNumber in enumerate(sweepList):
            abf.setSweep(sweepNumber)
            
            spike_in_sweep = (spiketimes[spiketimes[:,1]==int(sweepNumber+1)])[:,0]
            i1, i2 = int(abf.dataRate * lowerlim), int(abf.dataRate * upperlim) # plot part of the sweep
            dataX = abf.sweepX
            dataY = abf.sweepY
            colour = colors[c]
            sweepname = 'Sweep ' + str(sweepNumber)
            plt.plot(dataX, dataY, color=colour, alpha=1, lw=1, label=sweepname)
            
            plt.scatter(dataX[spike_in_sweep[:]], dataY[spike_in_sweep[:]], color=colour, marker='x')
           
        

        plt.xlim(abf.sweepX[i1], abf.sweepX[i2])
        plt.legend()
        
        plt.savefig(abf.abfID +'.png', dpi=600)
        plt.pause(0.05)
   except:
        print('plot failed')

def build_running_bin(array, time, start, end, bin=20, time_units='s', kind='nearest'):
    if time_units == 's':
        start = start * 1000
        end = end* 1000

        time = time*1000
    time_bins = np.arange(start, end+bin, bin)
    binned_ = np.full(time_bins.shape[0], np.nan, dtype=np.float64)
    index_ = np.digitize(time, time_bins)
    uni_index_ = np.unique(index_)
    for time_ind in uni_index_:
        data = np.asarray(array[index_==time_ind])
        data = np.nanmean(data)
        binned_[time_ind] = data
    nans = np.isnan(binned_)
    if np.any(nans):
        if time.shape[0] > 1:
            f = interpolate.interp1d(time, array, kind=kind, fill_value="extrapolate")
            new_data = f(time_bins)
            binned_[nans] = new_data[nans]
        else:
            binned_[nans] = np.nanmean(array)
    return binned_, time_bins




    
  # except:
    #    print('plot_failed')
debugplot = 0
running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width']
dfs = pd.DataFrame()
df_spike_count = pd.DataFrame()
df_running_avg_count = pd.DataFrame()
for root,dir,fileList in os.walk(files):
 for filename in fileList:
    if filename.endswith(".abf"):
        file_path = os.path.join(root,filename)
        try:
            abf = pyabf.ABF(file_path)
        
            if abf.sweepLabelY != 'Clamp Current (pA)' and abf.protocol != 'Gap free' and protocol_name in abf.protocol:
                print(filename + ' import')
              
                
            
                if bfeatcon == True:

                   df_running_avg_count = df_running_avg_count.append(temp_running_bin)
                   df_spike_count = df_spike_count.append(temp_spike_df, sort=True)

                   dfs = dfs.append(df, sort=True)
              #except:
               # print('Issue Processing ' + filename)

            else:
                print('Not correct protocol: ' + abf.protocol)
        except:
            print('Issue Processing ' + filename)

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