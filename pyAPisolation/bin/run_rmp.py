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

print("Load finished")

root = tk.Tk()
root.withdraw()
files = filedialog.askdirectory(
                                   title='Select dir File'
                                   )
root_fold = files

def crop_ap(abf):
    spikext = feature_extractor.SpikeFeatureExtractor(filter=0, dv_cutoff=20)
    dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
    spike_in_sweep = spikext.process(dataT, dataV, dataI)
    sweep_indi = np.arange(0, dataV.shape[0])
    if spike_in_sweep.empty == False:
        ap_start_ = spike_in_sweep['threshold_index'].to_numpy()
        ap_end_ = spike_in_sweep['trough_index'].to_numpy() + 300
        pairs = np.vstack((ap_start_, ap_end_)).T
        pair_data = []
        for p in pairs:
            temp = np.arange(p[0], p[1]).astype(np.int)
            pair_data.append(temp.tolist())
        pair_data = np.hstack(pair_data)
        pair_data = pair_data[pair_data<dataV.shape[0]]
        dataV[pair_data] = np.nan
        sweep_data = dataV
    else:
        sweep_data = abf.sweepY
    
        
    return sweep_data



def rmp_abf(abf, time=30, crop=True):
 #try:
    
    sweepsdata = []
    
            
    for sweepNumber in abf.sweepList:
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
        delta_vm = f_vm - e_vm
        sweep_time = abf.sweepLengthSec
        if abf.sweepLengthSec >= time:
            f60 = abf.dataPointsPerMs * int((time) * 1000)
            median_vm_last = np.nanmedian(abf.sweepY[-t1:])
            mode_vm_last = mode(abf.sweepY[-t1:], nan_policy='omit')[0][0]
        else:
            
            mode_vm_last = mode_vm
            median_vm_last= np.nanmedian(abf.sweepY)
        if mean_vm < -20 and mean_vm >-100:
            sweepsdata.append(np.hstack((mean_vm, std_vm, f_vm, median_vm, mode_vm, e_vm, median_vm_last, mode_vm_last, delta_vm, sweep_time)))
    sweep_full = np.vstack(sweepsdata)
    df = pd.DataFrame(data=sweep_full, columns=[f'Overall Mean vm','Overall STD vm', f'first {time}s Mean Vm', f'first {time}s Median Vm',f'first {time}s Mode Vm',  f'End {time}s Mean Vm', f'End {time}s median Vm', f'End {time}s mode Vm', 'Delta Vm', 'Length(s)'])
    df['fold_name'] = np.full(sweep_full.shape[0], abf.abfFolderPath)
    df['sweep number'] = abf.sweepList
    df['cell_name'] = np.full(sweep_full.shape[0], abf.abfID)
    return df
 
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

print("protocols")
for i, x in enumerate(protocol_n):
    print(str(i) + '. '+ str(x))
proto = input("enter Protocol to analyze: ")
try: 
    proto = int(proto)
except:
    proto = 0
protocol_name = protocol_n[proto]
lowerlim = input("Enter the time to analyze rmp (eg. first and last 10s)[in s]: ")

try: 
    lowerlim = np.float(lowerlim)
except:
    lowerlim  = 10


crop = input("[Experimental] Try to 'crop' out action potentials when analyzing RMP? (y/n): ")

try: 
    crop = str(crop)
    if crop == 'y' or crop == 'Y' or crop == 'yes':
        bcrop = True
    else:
        bcrop = False
except:
    bcrop = False

full_df = pd.DataFrame()
for root,dirs,fileList in os.walk(root_fold): 
    for x in fileList:
        fp = os.path.join(root, x)
        if '.abf' in x:
            try:
                abf = pyabf.ABF(fp)
                if abf.sweepLabelY != 'Clamp Current (pA)' and protocol_name in abf.protocol:
                    print(abf.abfID + ' import')
                    temp_df = rmp_abf(abf, lowerlim, bcrop)
                    if temp_df.empty == False:
                        full_df = full_df.append(temp_df)
            except:
              print('error processing file ' + fp)


with pd.ExcelWriter(root_fold + '/RMP_' + tag + '.xlsx') as runf:
        full_df.to_excel(runf, sheet_name="sweepwise RMP")
        full_df.groupby(['cell_name']).mean().to_excel(runf, sheet_name="Mean RMP")

print("==== SUCCESS ====")
input('Press ENTER to exit')
