
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
from scipy import stats
from ipfx import feature_extractor
from ipfx import subthresh_features as subt
from ipfx import feature_vectors as fv
from ipfx.sweep import Sweep
from sklearn.preprocessing import minmax_scale
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
        

def exp_grow(t, a, b, alpha):
    return a - b * np.exp(-alpha * t)

def exp_grow_2p(t, a, b1, alphaFast, b2, alphaSlow):
    return a - b1 * np.exp(-alphaFast * t) - b2*np.exp(-alphaSlow*t) 


def exp_decay_2p(t, a, b1, alphaFast, b2, alphaSlow):
    return a + b1*np.exp(-alphaFast*t) + b2*np.exp(-alphaSlow*t)
def exp_decay_1p(t, a, b1, alphaFast):
    return a + b1*np.exp(-alphaFast*t)
def exp_growth_factor(dataT,dataV,dataI, end_index=300):
    #try:
        
        diff_I = np.diff(dataI)
        upwardinfl = np.argmax(diff_I)

        #Compute out -50 ms from threshold
        dt = dataT[1] - dataT[0]
        offset = 0.05/ dt 

        end_index = int(end_index - offset)


        
        upperC = np.amax(dataV[upwardinfl:end_index])
        lowerC  = np.amin(dataV[upwardinfl:end_index])
        diffC = np.abs(lowerC - upperC) + 5
        t1 = dataT[upwardinfl:end_index] - dataT[upwardinfl]
        curve = curve_fit(exp_grow, t1, dataV[upwardinfl:end_index], maxfev=50000, bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))[0]
        curve2 = curve_fit(exp_grow_2p, t1, dataV[upwardinfl:end_index], maxfev=50000,   bounds=([-np.inf,  0, -np.inf,  0, -np.inf], [upperC + 5, diffC, np.inf, np.inf, np.inf]), xtol=None, method='trf')[0]
        tau = curve[2]
        plt.plot(t1, dataV[upwardinfl:end_index])
        plt.plot(t1, exp_grow_2p(t1, *curve2))
        plt.title(f" CELL will tau1 {1/curve2[2]} and tau2 {1/curve2[4]}, a {curve2[0]} and b1 {curve2[1]}, b2 {curve2[3]}")
        plt.pause(5)
        return 1/tau
    #except:
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


def find_zero(realC):
    #expects 1d array
    zero_ind = np.where(realC == 0)[0]
    ##Account for time constant?
    diff = np.diff(zero_ind)
    diff_jump = np.where(diff>2)[0][0]
    if diff_jump + 3000 > realC.shape[0]:
        _hop = diff_jump
    else:
        _hop = diff_jump + 3000

    zero_ind_crop = np.hstack((zero_ind[:diff_jump], zero_ind[_hop:]))
    return zero_ind_crop

def compute_vm_drift(realY, zero_ind):
    sweep_wise_mean = np.mean(realY[:,zero_ind], axis=1)
    mean_drift = np.abs(np.amax(sweep_wise_mean) - np.amin(sweep_wise_mean))
    abs_drift = np.abs(np.amax(realY[:,zero_ind], axis=1) - np.amin(realY[:,zero_ind], axis=1))

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
    try:
        zero_ind = find_zero(realC[0,:])
        mean_rms, max_rms = compute_rms(realY, zero_ind)
        mean_drift, max_drift = compute_vm_drift(realY, zero_ind)
        return [mean_rms, max_rms, mean_drift, max_drift]
    except:
        print("Failed to run QC on cell")
        return [np.nan, np.nan, np.nan, np.nan]

def compute_norm(realY):
     norm_y = minmax_scale(realY, axis=0)
     return norm_y

def compute_ap_vm(realX, realY, strt, end):
    ap = realY[strt:end]
    x_diff = np.diff(realX[strt:end]) * 1000
    ap_dv = np.diff(ap) / x_diff

    return ap, ap_dv

def downsample_array(a, size, method="resample"):
    if method=='avg':
        current_size = a.shape[0]
        window_width = int(np.ceil(current_size / size))
        avg = np.nanmean(a.reshape(-1, window_width), axis=1)
    elif method=="resample":
        avg = signal.resample(a, size)
    return avg

def equal_list_array(data):
    max_len = len(max(data,key=len))
    min_len = len(min(data,key=len))
    equalized = False
    for a, el in enumerate(data):
            len_fill = min_len - len(el)
            if len_fill==0:
                continue
            else:
                equalized = True
                remainder = np.remainder(len(el), min_len)
                if remainder == 0:
                    data[a] = downsample_array(el, min_len, method='avg')
                else:
                    data[a] = downsample_array(el, min_len, method='resample')

    nudata = np.vstack(data[:])
    return nudata, equalized

  # except:
    #    print('plot_failed')
debugplot = 0
running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width']

full_neuron_array = []
full_neuron_path = []

for root,dir,fileList in os.walk(files):
 for filename in fileList:
    if filename.endswith(".abf"):
            file_path = os.path.join(root,filename)
        #try:
            abf = pyabf.ABF(file_path)
        
            if abf.sweepLabelY != 'Clamp Current (pA)' and protocol_name in abf.protocol and abf.sweepCount==15:
                print(filename + ' import')
                np.nan_to_num(abf.data, nan=-9999, copy=False)
                if savfilter >0:
                    abf.data = signal.savgol_filter(abf.data, savfilter, polyorder=3)
                try:
                    del spikext
                except:
                     _ = 1
                 #If there is more than one sweep, we need to ensure we dont iterate out of range
                if abf.sweepCount > 1:
                    sweepcount = (abf.sweepCount)
                else:
                    sweepcount = 1
                df = pd.DataFrame()
                #Now we walk through the sweeps looking for action potentials
                neuron_data = {}
                neuron_data['ids'] = filename
                neuron_spike_count = []
                temp_spike_dfs = []
                temp_spike_dfs_nonzero = []
                temp_spiket_dfs = []
                full_dataI = []
                full_dataV = []
                sweepwise_latency = np.full(sweepcount, 0, dtype=np.float32)
                sweepwise_adapt = np.full(sweepcount, 0, dtype=np.float32)
                sweepwise_trough_averge = []
                sweepwise_peak_average = []
                sweepwise_ratio = []
                sweepwise_threshold = []
                sweepwise_width = []
                step_subt = []
                for sweepNumber in range(0, sweepcount): 
                    real_sweep_length = abf.sweepLengthSec - 0.0001
                    if sweepNumber < 9:
                        real_sweep_number = '00' + str(sweepNumber + 1)
                    elif sweepNumber > 8 and sweepNumber < 99:
                        real_sweep_number = '0' + str(sweepNumber + 1)
                    if lowerlim == 0 and upperlim == 0:
                        upperlim = real_sweep_length
                    elif upperlim > real_sweep_length:
                        upperlim = real_sweep_length
                    abf.setSweep(sweepNumber)
                    spikext = feature_extractor.SpikeFeatureExtractor(filter=filter, dv_cutoff=dv_cut, start=lowerlim, end=upperlim, max_interval=tp_cut,min_height=min_cut, min_peak=min_peak, thresh_frac=percent)
                    spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=lowerlim, end=upperlim)
                    dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
                    dt = (dataT[1] - dataT[0]) * 1000
                    full_dataI.append(dataI)
                    full_dataV.append(dataV)
                    if dataI.shape[0] < dataV.shape[0]:
                        dataI = np.hstack((dataI, np.full(dataV.shape[0] - dataI.shape[0], 0)))
                    spike_in_sweep = spikext.process(dataT, dataV, dataI)
                    spike_train = spiketxt.process(dataT, dataV, dataI, spike_in_sweep)
                    spike_count = spike_in_sweep.shape[0]
                    neuron_spike_count.append(spike_count)
                    temp_spike_dfs.append(spike_in_sweep)
                    temp_spiket_dfs.append(spike_train)

                    #compute sweep number specific features
                    if sweepNumber == 0:
                        neuron_data["subthresh_norm"] = compute_norm(downsample_array(dataV, 2000))
                        step_subt = np.hstack((step_subt, downsample_array(dataV, 2000)))
                    elif sweepNumber ==1:
                        #Take one prior 
                        neuron_data["subthresh_depol_norm"] = compute_norm(downsample_array(dataV, 2000))
                        step_subt = np.hstack((step_subt, downsample_array(dataV, 2000)))
                    elif sweepNumber == 3:
                        step_subt = np.hstack((step_subt, downsample_array(dataV, 2000)))


                    if spike_in_sweep.empty == False:
                        temp_spike_dfs_nonzero.append(spike_in_sweep)
                        trough_average,_  = build_running_bin(spike_in_sweep['fast_trough_v'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=lowerlim, end=upperlim)
                        peak_average = build_running_bin(spike_in_sweep['peak_v'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=lowerlim, end=upperlim)[0]
                        peak_max_rise = build_running_bin(spike_in_sweep['upstroke'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=lowerlim, end=upperlim)[0]
                        peak_max_down = build_running_bin(spike_in_sweep['downstroke'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=lowerlim, end=upperlim)[0]
                        threshold = build_running_bin(spike_in_sweep['downstroke'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=lowerlim, end=upperlim)[0]
                        peak_width = build_running_bin(spike_in_sweep['width'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=lowerlim, end=upperlim)[0]
                        ratio = np.abs(peak_max_rise / peak_max_down)
                        sweepwise_trough_averge.append(trough_average)
                        sweepwise_peak_average.append(peak_average)
                        sweepwise_ratio.append(ratio)
                        sweepwise_threshold.append(threshold)
                        sweepwise_width.append(peak_width)
                    if 'latency' in spike_train.keys():
                        sweepwise_latency[sweepNumber] = spike_train['latency']
                    if 'adapt' in spike_train.keys():
                        if np.isnan(spike_train['adapt']) == False:
                            sweepwise_adapt[sweepNumber] = spike_train['adapt']
                neuron_spike_count = np.array(neuron_spike_count)
                rheobase_sweep = np.nonzero(neuron_spike_count)[0][0]
                
                #Grab first AP V - D/V
                first_spike_df = temp_spike_dfs[rheobase_sweep]
                first_spike_start = first_spike_df['threshold_index'].to_numpy()[0]
                time_aft = 10 / dt #grab 10 ms after
                first_spike_end = np.int(first_spike_start + time_aft)
                abf.setSweep(rheobase_sweep)
                dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
                neuron_data["first_ap_v"], neuron_data["first_ap_dv"] = compute_ap_vm(dataT, dataV, int(first_spike_start), int(first_spike_end))


                #Sweep with at least 5 aps for isi_shape
                isi_sweep = np.argmin(np.abs(neuron_spike_count-5))
                abf.setSweep(isi_sweep)
                dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
                ipfx_sweep = Sweep(dataT, dataV, dataI, "CurrentClamp", (1/(dt/1000)))
                isi_norm = fv.isi_shape(ipfx_sweep, temp_spike_dfs[isi_sweep], upperlim)
                neuron_data["isi_shape"] = isi_norm

            

                #build inst_freq
                inst_freq = fv.inst_freq_vector(temp_spike_dfs_nonzero, lowerlim, upperlim)
                psth = fv.psth_vector(temp_spike_dfs_nonzero, lowerlim, upperlim)
                neuron_data['inst_freq'] = inst_freq
                neuron_data['psth'] = psth
                #Build the other features
                len_fv_sweep = len(sweepwise_trough_averge[0])
                blank_non_spike = np.full(len_fv_sweep, 0)
                non_spike_fv = np.hstack([blank_non_spike for x in neuron_spike_count[neuron_spike_count==0]])
                neuron_data["spiking_fast_trough_v"] = np.hstack((non_spike_fv, np.hstack(sweepwise_trough_averge)))
                neuron_data["spiking_peak_v"] = np.hstack((non_spike_fv, np.hstack(sweepwise_peak_average)))
                neuron_data["spiking_threshold_v"] = np.hstack((non_spike_fv, np.hstack(sweepwise_threshold)))
                neuron_data["spiking_upstroke_downstroke_ratio"] = np.hstack((non_spike_fv, np.hstack(sweepwise_ratio)))
                neuron_data["spiking_width"]  = np.hstack((non_spike_fv, np.hstack(sweepwise_width)))

                neuron_data['latency'] = sweepwise_latency
                neuron_data['adapt'] = sweepwise_adapt
                neuron_data['step_subthresh'] = np.hstack(step_subt)
                neuron_data['FI'] = neuron_spike_count / .7
                print("Processing Complete")
                
                full_neuron_path.append(file_path)
                full_neuron_array.append(neuron_data)
            else:
                print('Not correct protocol: ' + abf.protocol)
        #except:
          # print('Issue Processing ' + filename)


#Go through keys and stack files togethers
data_keys = full_neuron_array[0].keys()
for key in data_keys:
    print(f"processing {key}")
    temp_array =[]
    for row in full_neuron_array:
        temp_data = row[key]
        temp_array.append(temp_data)
    if key!='ids':
        npy_arr, eql_b = equal_list_array(temp_array)
        if eql_b == True:
            print(f"{key} had uneven data lengths")
        np.savetxt(key+".csv", npy_arr, fmt='%.18f', delimiter=',')
    else:
        np.savetxt(key+".csv", temp_array, fmt='%.18s', delimiter=',')


np.savetxt("neuron_files.csv", full_neuron_path, fmt='%.128s', delimiter=',')

print(f"Ran analysis with, dVdt thresh: {dv_cut}mV/s, thresh to peak max: {tp_cut}s, thresh to peak min height: {min_cut}mV, and min peak voltage: {min_peak}mV")

  

print("==== SUCCESS ====")
input('Press ENTER to exit')