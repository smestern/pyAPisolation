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
        mmode_vm = mode(data, nan_policy='omit')[0][0]
        mean_vm = mmode_vm
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
        #if mean_vm < -20 and mean_vm >-100:
        sweepsdata.append(np.hstack((mean_vm, std_vm, f_vm, median_vm, mode_vm, e_vm, median_vm_last, mode_vm_last, delta_vm, sweep_time)))
    sweep_full = np.vstack(sweepsdata)
    df = pd.DataFrame(data=sweep_full, columns=[f'Overall Mean vm','Overall STD vm', f'first {time}s Mean Vm', f'first {time}s Median Vm',f'first {time}s Mode Vm',  f'End {time}s Mean Vm', f'End {time}s median Vm', f'End {time}s mode Vm', 'Delta Vm', 'Length(s)'])
    df['fold_name'] = np.full(sweep_full.shape[0], abf.abfFolderPath)
    df['sweep number'] = abf.sweepList
    df['cell_name'] = np.full(sweep_full.shape[0], abf.abfID)
    return df
 
 #except:
     #return pd.DataFrame

 
def find_zero(realC):
    #expects 1d array
    zero_ind = np.where(realC == 0)[0]
    ##Account for time constant?
    diff = np.diff(zero_ind)
    if np.amax(diff) > 1:
        diff_jump = np.where(diff>2)[0][0]
        if diff_jump + 3000 > realC.shape[0]:
            _hop = diff_jump
        else:
            _hop = diff_jump + 3000

        zero_ind_crop = np.hstack((zero_ind[:diff_jump], zero_ind[_hop:]))
    else: 
        zero_ind_crop = zero_ind
    return zero_ind_crop

def compute_vm_drift(realY, zero_ind):
    sweep_wise_mean = np.mean(realY[:,zero_ind], axis=1)
    mean_drift = np.abs(np.amax(sweep_wise_mean) - np.amin(sweep_wise_mean))
    abs_drift = np.abs(np.amax(realY[:,zero_ind]) - np.amin(realY[:,zero_ind]))

    return mean_drift, abs_drift


def compute_rms(realY, zero_ind):
    mean = np.mean(realY[:,zero_ind], axis=1)
    rms = []
    for x in np.arange(mean.shape[0]):
        temp = np.sqrt(np.mean(np.square(realY[x,zero_ind] - mean[x])))
        rms = np.hstack((rms, temp))
    full_mean = np.mean(rms)
    return full_mean, np.amax(rms)

def run_qc(realY, realC):
    #try:
        zero_ind = find_zero(realC[0,:])
        mean_rms, max_rms = compute_rms(realY, zero_ind)
        mean_drift, max_drift = compute_vm_drift(realY, zero_ind)
        return [mean_rms, max_rms, mean_drift, max_drift]
    #except:
       # print("Failed to run QC on cell")
        return [np.nan, np.nan, np.nan, np.nan]





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
for root,dirs,fileList in os.walk(root_fold): 
    for x in fileList:
        fp = os.path.join(root, x)
        if '.abf' in x:
           try:
                abf = pyabf.ABF(fp)
                print(f"opening {abf.abfID}")
                if abf.sweepLabelY != 'Clamp Current (pA)':
                    temp_spike_df = pd.DataFrame()
                    temp_spike_df['filename'] = [abf.abfID]
                    temp_spike_df['foldername'] = [os.path.dirname(fp)]
                    temp_spike_df['__a_protocol'] = [abf.protocol]
                    abf.setSweep(0)
                    full_dataV = abf.sweepY
                    full_dataI = abf.sweepC
                    for sweep in abf.sweepList:
                        abf.setSweep(sweep)
                        full_dataV = np.vstack((full_dataV, abf.sweepY))
                        full_dataI = np.vstack((full_dataI, abf.sweepC))
                    QC_Vars = run_qc(full_dataV, full_dataI)
                    rmp_df = rmp_abf(abf, 1, False)
                    
                    temp_spike_df["Mode RMP"] = mode(rmp_df['Overall Mean vm'].to_numpy(), nan_policy='omit')[0][0]
                    temp_spike_df["QC - MEAN RMS"] = [QC_Vars[0]]
                    temp_spike_df["QC - MAX RMS"] = [QC_Vars[1]]
                    temp_spike_df["QC - MEAN VM DRIFT"] = [QC_Vars[2]]
                    temp_spike_df["QC - MAX VM DRIFT"] = [QC_Vars[3]]
                    if temp_spike_df.empty == False:
                        full_df = full_df.append(temp_spike_df)
           except:
               print('error processing file ' + fp)


with pd.ExcelWriter(root_fold + '/QC_' + tag + '.xlsx') as runf:
        cols = full_df.columns.values
        df_ind = full_df.loc[:,cols[[1,0]]]
        index = pd.MultiIndex.from_frame(df_ind)
        full_df.set_index(index).to_excel(runf, sheet_name='Full QC')
        folder = full_df['foldername'].to_numpy()
        full_df = full_df.iloc[:,3:]
        cell_wise = full_df.groupby(folder, as_index=True)
        cell_wise_diff = cell_wise
        cell_wise_diff.diff().to_excel(runf, sheet_name='Mode')
        print()
        
print("==== SUCCESS ====")
input('Press ENTER to exit')
