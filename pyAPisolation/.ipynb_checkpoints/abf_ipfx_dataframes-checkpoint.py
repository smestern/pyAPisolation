


import sys
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
from scipy.optimize import curve_fit
from ipfx import feature_extractor
from ipfx import subthresh_features as subt
import pyabf
from patch_utils import *
from patch_subthres import *


running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width']

def analyze_abf(abf, sweeplist=None, plot=-1, param_dict=None):
        np.nan_to_num(abf.data, nan=-9999, copy=False)
        #If there is more than one sweep, we need to ensure we dont iterate out of range
        if sweeplist == None:
            if abf.sweepCount > 1:
                sweepcount = abf.sweepList
            else:
                sweepcount = [0]
        df = pd.DataFrame()
        #Now we walk through the sweeps looking for action potentials
        temp_spike_df = pd.DataFrame()
        temp_spike_df['filename'] = [abf.abfID]
        temp_spike_df['foldername'] = [os.path.dirname(abf.abfFilePath)]
        temp_running_bin = pd.DataFrame()
                
        for sweepNumber in sweepcount: 
            real_sweep_length = abf.sweepLengthSec - 0.0001
            if sweepNumber < 9:
                real_sweep_number = '00' + str(sweepNumber + 1)
            elif sweepNumber > 8 and sweepNumber < 99:
                real_sweep_number = '0' + str(sweepNumber + 1)
            if param_dict['start'] == 0 and param_dict['end'] == 0: 
                param_dict['end']= real_sweep_length
            elif param_dict['end'] > real_sweep_length:
                param_dict['end'] = real_sweep_length
            ##Look for a subthreshold component
            spike_in_sweep, spike_train = analyze_spike_sweep(abf, sweepNumber, param_dict) ### Returns the default Dataframe Returned by 
            temp_spike_df, df, temp_running_bin = _build_sweepwise_dataframe(abf, real_sweep_number, spike_in_sweep, spike_train, temp_spike_df, df, temp_running_bin, param_dict)
        temp_spike_df, df, temp_running_bin = _build_full_df(abf, temp_spike_df, df, temp_running_bin, sweepcount)
        spiketimes = np.transpose(np.vstack((np.ravel(df['peak_index'].to_numpy()), np.ravel(df['sweep Number'].to_numpy()))))
        plotabf(abf, spiketimes, param_dict['start'], param_dict['end'], plot)
        return temp_spike_df, df, temp_running_bin

                


def analyze_spike_sweep(abf, sweepNumber, param_dict):
    abf.setSweep(sweepNumber)
    spikext = feature_extractor.SpikeFeatureExtractor(**param_dict)
    spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=param_dict['start'], end=param_dict['end'])
    dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
    if dataI.shape[0] < dataV.shape[0]:
                dataI = np.hstack((dataI, np.full(dataV.shape[0] - dataI.shape[0], 0)))
    spike_in_sweep = spikext.process(dataT, dataV, dataI)
    spike_train = spiketxt.process(dataT, dataV, dataI, spike_in_sweep)
    return spike_in_sweep, spike_train

def _build_sweepwise_dataframe(abf, real_sweep_number, spike_in_sweep, spike_train, temp_spike_df, df, temp_running_bin, param_dict):
            try:
               
                uindex = abf.epochPoints[1] + 1
            except:
                uindex = 0
            
            spike_count = spike_in_sweep.shape[0]
            temp_spike_df["Sweep " + real_sweep_number + " spike count"] = [spike_count]
            current_str = np.array2string(np.unique(abf.sweepC))
            current_str = current_str.replace('[', '')
            current_str = current_str.replace(' 0.', '')
            current_str = current_str.replace(']', '')
            sweep_running_bin = pd.DataFrame()
            temp_spike_df["Current_Sweep " + real_sweep_number + " current injection"] = [current_str]
            time_bins = np.arange(param_dict['start']*1000, param_dict['end']*1000+20, 20)
            _run_labels = []
            for p in running_lab:
                    temp_lab = []
                    for x in time_bins :
                        temp_lab = np.hstack((temp_lab, f'{p} {x} bin AVG'))
                    _run_labels.append(temp_lab)
            _run_labels = np.hstack(_run_labels).tolist()
            nan_row_run = np.ravel(np.full((5, time_bins.shape[0]), np.nan)).reshape(1,-1)
            try:
                temp_spike_df['baseline voltage' + real_sweep_number] = subt.baseline_voltage(abf.sweepX, abf.sweepY, start=0.1, filter_frequency=param_dict['filter'])
            except:
                print('Fail to find baseline voltage')
                    

            if spike_count > 0:
                temp_spike_df["isi_Sweep " + real_sweep_number + " isi"] = [spike_train['first_isi']]
                trough_averge,_  = build_running_bin(spike_in_sweep['fast_trough_v'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])
                peak_average = build_running_bin(spike_in_sweep['peak_v'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
                peak_max_rise = build_running_bin(spike_in_sweep['upstroke'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
                peak_max_down = build_running_bin(spike_in_sweep['downstroke'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
                peak_width = build_running_bin(spike_in_sweep['width'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
                        
                sweep_running_bin = pd.DataFrame(data=np.hstack((trough_averge, peak_average, peak_max_rise, peak_max_down, peak_width)).reshape(1,-1), columns=_run_labels, index=[real_sweep_number])
                spike_train_df = pd.DataFrame(spike_train, index=[0])
                nan_series = pd.DataFrame(np.full(abs(spike_count-1), np.nan))
                #spike_train_df = spike_train_df.append(nan_series)
                spike_in_sweep['spike count'] = np.hstack((spike_count, np.full(abs(spike_count-1), np.nan)))
                spike_in_sweep['sweep Number'] = np.full(abs(spike_count), int(real_sweep_number))
                temp_spike_df["spike_amp" + real_sweep_number + " 1"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[0] - spike_in_sweep['threshold_v'].to_numpy()[0])
                temp_spike_df["spike_thres" + real_sweep_number + " 1"] = spike_in_sweep['threshold_v'].to_numpy()[0]
                temp_spike_df["spike_peak" + real_sweep_number + " 1"] = spike_in_sweep['peak_v'].to_numpy()[0]
                temp_spike_df["spike_rise" + real_sweep_number + " 1"] = spike_in_sweep['upstroke'].to_numpy()[0]
                temp_spike_df["spike_decay" + real_sweep_number + " 1"] = spike_in_sweep['downstroke'].to_numpy()[0]
                temp_spike_df["spike_AHP 1" + real_sweep_number + " "] = spike_in_sweep['fast_trough_v'].to_numpy()[0]
                temp_spike_df["spike_AHP height 1" + real_sweep_number + " "] = abs(spike_in_sweep['peak_v'].to_numpy()[0] - spike_in_sweep['fast_trough_v'].to_numpy()[0])
                temp_spike_df["latency_" + real_sweep_number + " latency"] = spike_train['latency']
                temp_spike_df["spike_width" + real_sweep_number + "1"] = spike_in_sweep['width'].to_numpy()[0]
                        
                #temp_spike_df["exp growth" + real_sweep_number] = [exp_growth_factor(dataT, dataV, dataI, spike_in_sweep['threshold_index'].to_numpy()[0])]
                        
                if spike_count > 2:
                    f_isi = spike_in_sweep['peak_t'].to_numpy()[-1]
                    l_isi = spike_in_sweep['peak_t'].to_numpy()[-2]
                    temp_spike_df["last_isi" + real_sweep_number + " isi"] = [abs( f_isi- l_isi )]
                    spike_in_sweep['isi_'] = np.hstack((np.diff(spike_in_sweep['peak_t'].to_numpy()), np.nan))
                    temp_spike_df["min_isi" + real_sweep_number + " isi"] = np.nanmin(np.hstack((np.diff(spike_in_sweep['peak_t'].to_numpy()), np.nan)))
                    #temp_spike_df["spike_" + real_sweep_number + " 2"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[1] - spike_in_sweep['threshold_v'].to_numpy()[1])
                    #temp_spike_df["spike_" + real_sweep_number + " 3"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[-1] - spike_in_sweep['threshold_v'].to_numpy()[-1])
                    #temp_spike_df["spike_" + real_sweep_number + "AHP 2"] = spike_in_sweep['fast_trough_v'].to_numpy()[1]
                    #temp_spike_df["spike_" + real_sweep_number + "AHP 3"] = spike_in_sweep['fast_trough_v'].to_numpy()[-1]
                    #temp_spike_df["spike_" + real_sweep_number + "AHP height 2"] = abs(spike_in_sweep['peak_v'].to_numpy()[1] - spike_in_sweep['fast_trough_v'].to_numpy()[1])
                    #temp_spike_df["spike_" + real_sweep_number + "AHP height 3"] = abs(spike_in_sweep['peak_v'].to_numpy()[-1] - spike_in_sweep['fast_trough_v'].to_numpy()[-1])
                    #temp_spike_df["spike_width" + real_sweep_number + "2"] = spike_in_sweep['width'].to_numpy()[1]
                    #temp_spike_df["spike_width" + real_sweep_number + "3"] = spike_in_sweep['width'].to_numpy()[-1]
                else:
                    temp_spike_df["last_isi" + real_sweep_number + " isi"] = [np.nan]
                    spike_in_sweep['isi_'] = np.hstack((np.full(abs(spike_count), np.nan)))
                    temp_spike_df["min_isi" + real_sweep_number + " isi"] = [spike_train['first_isi']]
                spike_in_sweep = spike_in_sweep.join(spike_train_df)
                print("Processed Sweep " + str(real_sweep_number) + " with " + str(spike_count) + " aps")
                df = df.append(spike_in_sweep, ignore_index=True, sort=True)
            else:
                temp_spike_df["latency_" + real_sweep_number + " latency"] = [np.nan]
                temp_spike_df["isi_Sweep " + real_sweep_number + " isi"] = [np.nan]
                temp_spike_df["last_isi" + real_sweep_number + " isi"] = [np.nan]
                temp_spike_df["spike_amp" + real_sweep_number + " 1"] = [np.nan]
                temp_spike_df["spike_thres" + real_sweep_number + " 1"] = [np.nan]
                temp_spike_df["spike_rise" + real_sweep_number + " 1"] = [np.nan]
                temp_spike_df["spike_decay" + real_sweep_number + " 1"] = [np.nan]
                #temp_spike_df["spike_" + real_sweep_number + " 2"] = [np.nan]
                #temp_spike_df["spike_" + real_sweep_number + " 3"] = [np.nan]
                temp_spike_df["spike_AHP 1" + real_sweep_number + " "] = [np.nan]
                temp_spike_df["spike_peak" + real_sweep_number + " 1"] = [np.nan]
                temp_spike_df["spike_AHP height 1" + real_sweep_number + " "] = [np.nan]
                #temp_spike_df["spike_" + real_sweep_number + "AHP 2"] = [np.nan]
                #temp_spike_df["spike_" + real_sweep_number + "AHP 3"] = [np.nan]
                #temp_spike_df["spike_" + real_sweep_number + "AHP height 2"] = [np.nan]
                #temp_spike_df["spike_" + real_sweep_number + "AHP height 3"] = [np.nan]
                temp_spike_df["latency_" + real_sweep_number + " latency"] = [np.nan]
                temp_spike_df["spike_width" + real_sweep_number + "1"] = [np.nan]
                #temp_spike_df["spike_width" + real_sweep_number + "2"] = [np.nan]
                #temp_spike_df["spike_width" + real_sweep_number + "3"] = [np.nan]
                temp_spike_df["min_isi" + real_sweep_number + " isi"] = [np.nan]
                sweep_running_bin = pd.DataFrame(data=nan_row_run, columns=_run_labels, index=[real_sweep_number])
                #temp_spike_df["exp growth" + real_sweep_number] = [np.nan]
            sweep_running_bin['Sweep Number'] = [real_sweep_number]
            sweep_running_bin['filename'] = [abf.abfID]
            sweep_running_bin['foldername'] = [os.path.dirname(abf.abfFilePath)]
            temp_running_bin = temp_running_bin.append(sweep_running_bin)
            return temp_spike_df, df, temp_running_bin


def _build_full_df(abf, temp_spike_df, df, temp_running_bin, sweepList):
        temp_spike_df['protocol'] = [abf.protocol]
        if df.empty:
            df = df.assign(__file_name=np.full(1,abf.abfID))
            df = df.assign(__fold_name=np.full(1,os.path.dirname(file_path)))
            print('no spikes found')
        else:
            df = df.assign(__file_name=np.full(len(df.index),abf.abfID))
            df = df.assign(__fold_name=np.full(len(df.index),os.path.dirname(abf.abfFilePath)))
            rheo_sweep = df['sweep Number'].to_numpy()[0]
            abf.setSweep(int(rheo_sweep - 1))
            rheobase_current = abf.sweepC[np.argmax(abf.sweepC)]
            temp_spike_df["rheobase_current"] = [rheobase_current]
                    
            temp_spike_df["rheobase_latency"] = [df['latency'].to_numpy()[0]]
            temp_spike_df["rheobase_thres"] = [df['threshold_v'].to_numpy()[0]]
            temp_spike_df["rheobase_width"] = [df['width'].to_numpy()[0]]
            temp_spike_df["rheobase_heightPT"] = [abs(df['peak_v'].to_numpy()[0] - df['fast_trough_v'].to_numpy()[0])]
            temp_spike_df["rheobase_heightTP"] = [abs(df['threshold_v'].to_numpy()[0] - df['peak_v'].to_numpy()[0])]
                
            temp_spike_df["rheobase_upstroke"] = [df['upstroke'].to_numpy()[0]]
            temp_spike_df["rheobase_downstroke"] = [df['upstroke'].to_numpy()[0]]
            temp_spike_df["rheobase_fast_trough"] = [df['fast_trough_v'].to_numpy()[0]]
            temp_spike_df["mean_current"] = [np.nanmean(df['peak_i'].to_numpy())]
            temp_spike_df["mean_latency"] = [np.nanmean(df['latency'].to_numpy())]
            temp_spike_df["mean_thres"] = [np.nanmean(df['threshold_v'].to_numpy())]
            temp_spike_df["mean_width"] = [np.nanmean(df['width'].to_numpy())]
            temp_spike_df["mean_heightPT"] = [np.nanmean(abs(df['peak_v'].to_numpy() - df['fast_trough_v'].to_numpy()))]
            temp_spike_df["mean_heightTP"] = [np.nanmean(abs(df['threshold_v'].to_numpy() - df['peak_v'].to_numpy()))]
            temp_spike_df["mean_upstroke"] = [np.nanmean(df['upstroke'].to_numpy())]
            temp_spike_df["mean_downstroke"] = [np.nanmean(df['upstroke'].to_numpy())]
            temp_spike_df["mean_fast_trough"] = [np.nanmean(df['fast_trough_v'].to_numpy())]
            spiketimes = np.transpose(np.vstack((np.ravel(df['peak_index'].to_numpy()), np.ravel(df['sweep Number'].to_numpy()))))
           

        full_dataI = []
        full_dataV = []
        for x in sweepList:
            abf.setSweep(x)
            full_dataI.append(abf.sweepC)
            full_dataV.append(abf.sweepY)
        full_dataI = np.vstack(full_dataI) 
        full_dataV = np.vstack(full_dataV) 
        decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, _ = exp_decay_factor(abf.sweepX, np.nanmean(full_dataV,axis=0), np.nanmean(full_dataI,axis=0), 3000, abf_id=abf.abfID)
                
        temp_spike_df["fast decay avg"] = [decay_fast]
        temp_spike_df["slow decay avg"] = [decay_slow]
        temp_spike_df["Curve fit A"] = [curve[0]]
        temp_spike_df["Curve fit b1"] = [curve[1]]
        temp_spike_df["Curve fit b2"] = [curve[3]]
        temp_spike_df["R squared 2 phase"] = [r_squared_2p]
        temp_spike_df["R squared 1 phase"] = [r_squared_1p]
        if r_squared_2p > r_squared_1p:
            temp_spike_df["Best Fit"] = [2]
        else:
            temp_spike_df["Best Fit"] = [1]
        return  temp_spike_df, df, temp_running_bin
                
def subthres_a():
    if dataI[np.argmin(dataI)] < 0:
                        try:
                            if lowerlim < 0.1:
                                b_lowerlim = 0.1
                            else:
                                b_lowerlim = 0.1
                            #temp_spike_df['baseline voltage' + real_sweep_number] = subt.baseline_voltage(dataT, dataV, start=b_lowerlim)
                            temp_spike_df['sag' + real_sweep_number] = subt.sag(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                            temp_spike_df['time_constant' + real_sweep_number] = subt.time_constant(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                            
                            #temp_spike_df['voltage_deflection' + real_sweep_number] = subt.voltage_deflection(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                        except:
                            print("Subthreshold Processing Error with " + str(abf.abfID))