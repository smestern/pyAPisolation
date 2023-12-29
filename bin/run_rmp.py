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
from scipy.stats import mode
from ipfx import feature_extractor
from ipfx import subthresh_features as subt
import pyabf

from pyAPisolation.patch_utils import build_running_bin

print("Load finished")
DEBUG = False
root = tk.Tk()
root.withdraw()
files = filedialog.askdirectory(
                                   title='Select dir File'
                                   )
root_fold = files

def crop_ap(abf):
    print("Finding Spikes to be Removed")
    spikext = feature_extractor.SpikeFeatureExtractor(filter=0, dv_cutoff=20, thresh_frac=0.2)
    dataT, dataV, dataI = abf.sweepX, np.copy(abf.sweepY), np.copy(abf.sweepC)
    dt = dataT[1] - dataT[0]
    try:
        spike_in_sweep = spikext.process(dataT, dataV, dataI)
    except:
        spike_in_sweep = pd.DataFrame()
    sweep_indi = np.arange(0, dataV.shape[0])
    if spike_in_sweep.empty == False:
        #remove spikes
        print(f" === Found {spike_in_sweep.shape[0]} spikes === ")
        ap_start_ = spike_in_sweep['threshold_index'].to_numpy() - 500
        ap_end_ = spike_in_sweep['trough_index'].to_numpy() + 500
        pairs = np.vstack((ap_start_, ap_end_)).T
        pairs = np.nan_to_num(pairs, nan=len(dataT))
        pairs = pairs.astype(np.int)
        pair_data = []
        for p in pairs:
            if (p[1] - p[0])*dt > 0.1:
                print(f" === Found a long spike at {dataT[p[0]]} === ")
                p[1] = np.clip(p[1], p[0], p[0]+int(0.1/dt))
            
            #also enforce p[1] <= len()
            p[1] = np.clip(p[1], p[0], len(dataT)-1).astype(np.int)
            temp = np.arange(p[0], p[1]).astype(np.int)
            pair_data.append(temp.tolist())
            print(f" === cropping spike between {dataT[int(p[0])]} and {dataT[(p[1])]} === ")
        pair_data = np.hstack(pair_data)
        pair_data = pair_data[pair_data<dataV.shape[0]]
        dataV[pair_data] = np.nan
        sweep_data = dataV


        #debug plot
        if False:
            plt.plot(abf.sweepX, abf.sweepY, 'k')
            plt.plot(abf.sweepX, sweep_data, 'r')
            plt.scatter(abf.sweepX[pairs[:,0]], abf.sweepY[pairs[:,0]], c='g', s=100)
            plt.scatter(abf.sweepX[pairs[:,1]], abf.sweepY[pairs[:,1]], c='b', s=100)
            plt.show()
    else:
        sweep_data = abf.sweepY
    
    
        
    return sweep_data

def running_bin(x, y, bin_time):
    bin_x = np.arange(x[0], x[-1]+bin_time, step=bin_time)
    binned_indices  = np.digitize(x, bin_x)
    dict_running = {}
    for ind in np.unique(binned_indices):
        bool_true = (binned_indices==ind)
        dict_running[bin_x[ind]] = np.nanmean(y[bool_true])
    return pd.DataFrame.from_dict(dict_running, orient='index')


def rmp_abf(abf, time=30, crop=True, bin_time=100):
    sweepsdata = []
    running_sweeps = []
    for sweepNumber in abf.sweepList:
        print(f"Processing sweep number {sweepNumber}")
        #f10 = int((abf.sweepLengthSec * .10) * 1000)
        f10 = int((time) * 1000)
        t1 = abf.dataPointsPerMs * f10
        if t1 >= abf.sweepY.shape[0]:
            t1 = abf.sweepY.shape[0] - 1
        abf.setSweep(sweepNumber)
        if crop == True:
            data = crop_ap(abf)
        else:
            data = abf.sweepY
        mean_vm = np.nanmean(data)
        std_vm = np.nanstd(data)
        f_vm = np.nanmean(data[:t1])
        e_vm = np.nanmean(data[-t1:])
        median_vm = np.nanmedian(data[:t1])
        mode_vm = mode(data[:t1], nan_policy='omit')[0][0]

        #Compute the running bin
        #df_raw = pd.DataFrame(data=data, index=abf.sweepX)
        df_running = running_bin(abf.sweepX, data, bin_time/1000)
        running_sweeps.append(df_running)

        delta_vm = f_vm - e_vm
        sweep_time = abf.sweepLengthSec
        if abf.sweepLengthSec >= time:
            f60 = abf.dataPointsPerMs * int((time) * 1000)
            median_vm_last = np.nanmedian(abf.sweepY[-t1:])
            mode_vm_last = mode(abf.sweepY[-t1:], nan_policy='omit')[0][0]
        else:
            
            mode_vm_last = mode_vm
            median_vm_last= np.nanmedian(abf.sweepY)
        if mean_vm < 200 and mean_vm >-1000:
            sweepsdata.append(np.hstack((mean_vm, std_vm, f_vm, median_vm, mode_vm, e_vm, median_vm_last, mode_vm_last, delta_vm, sweep_time)))
    sweep_full = np.vstack(sweepsdata)
    df = pd.DataFrame(data=sweep_full, columns=[f'Overall Mean vm','Overall STD vm', f'first {time}s Mean Vm', f'first {time}s Median Vm',f'first {time}s Mode Vm',  f'End {time}s Mean Vm', f'End {time}s median Vm', f'End {time}s mode Vm', 'Delta Vm', 'Length(s)'])
    df['fold_name'] = np.full(sweep_full.shape[0], os.path.dirname(abf.abfFilePath))
    df['sweep number'] = abf.sweepList[:sweep_full.shape[0]]
    df['cell_name'] = np.full(sweep_full.shape[0], abf.abfID)

    df_running = pd.concat(running_sweeps, axis=0)
    return df, df_running
 
 #except:
     #return pd.DataFrame
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




print("protocols")
for i, x in enumerate(protocol_n):
    print(str(i) + '. '+ str(x))
proto = input("enter Protocol to analyze (enter -1 to not filter any protocol): ")
try: 
    proto = int(proto)
except:
    proto = -1
if proto == -1:
    protocol_name = ''
else:
    protocol_name = protocol_n[proto]


lowerlim = input("Enter the time to analyze rmp (eg. first and last 10s)[in s]: ")

try: 
    lowerlim = np.float32(lowerlim)
except:
    lowerlim  = 10

bin_time = input("Enter the bin size for building a running bin [in ms]: ")

try: 
    bin_time = np.float32(bin_time)
except:
    bin_time  = 100




crop = input("[Experimental] Try to 'crop' out action potentials when analyzing RMP? (y/n): ")

try: 
    crop = str(crop)
    if crop == 'y' or crop == 'Y' or crop == 'yes':
        bcrop = True
    else:
        bcrop = False
except:
    bcrop = False


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




full_df = pd.DataFrame()
full_df_running = pd.DataFrame()
for root,dirs,fileList in os.walk(root_fold): 
    
    for x in fileList:
        fp = os.path.join(root, x)
        if '.abf' in x:
            
            try:
                abf = pyabf.ABF(fp)
                if proto == -1 or protocol_name in abf.protocol:
                    print(abf.abfID + ' import')
                    temp_df, temp_df_running = rmp_abf(abf, lowerlim, bcrop, bin_time)
                    if temp_df.empty == False:
                        full_df = full_df.append(temp_df)
                        full_df_running = full_df_running.join(temp_df_running.rename({0: temp_df['cell_name'].to_numpy()[0]}, axis='columns'), how='outer')
            except:
                print('error processing file ' + fp)


with pd.ExcelWriter(root_fold + '/RMP_' + tag + '.xlsx') as runf:
        full_df.to_excel(runf, sheet_name="sweepwise RMP")
        full_df.groupby(['cell_name']).mean().to_excel(runf, sheet_name="Mean RMP")
        full_df_running.to_excel(runf, sheet_name="running bin RMP")

print("==== SUCCESS ====")
input('Press ENTER to exit')
