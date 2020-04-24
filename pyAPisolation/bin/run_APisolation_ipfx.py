
print("Loading...")
import sys
import numpy as np
from numpy import genfromtxt
import pyabf
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#import pyAPisolation as apis
from ipfx import feature_extractor
from ipfx import subthresh_features as subt
print("Load finished")
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
        #plt.clf()
        #plt.plot(t1, dataV[downwardinfl:end_index], label='Data')
        #plt.plot(t1, exp_decay_2p(t1, *curve), label='2 phase fit')
        #plt.plot(t1, exp_decay_1p(t1, curve[0], curve[1], curve[2]) + np.abs(upperC - np.amax(exp_decay_1p(t1, curve[0], curve[1], curve[2]))), label='Phase 1')
        #plt.plot(t1, exp_decay_1p(t1, curve[0], curve[3], curve[4]) + np.abs(upperC - np.amax(exp_decay_1p(t1, curve[0], curve[3], curve[4]))), label='Phase 2')
        #plt.legend()
        ##plt.pause(0.05)
        #plt.savefig(abf_id+'.png')
        #plt.close() 
        tau1 = 1/curve[2]
        tau2 = 1/curve[4]
        fast = np.min([tau1, tau2])
        slow = np.max([tau1, tau2])
        return tau1, tau2, curve, r_squared_2p, r_squared_1p
     except:
        return np.nan, np.nan, np.array([np.nan,np.nan,np.nan,np.nan,np.nan])



print('loading protocols...')
protocol = []
for root,dir,fileList in os.walk(files):
 for filename in fileList:
    if filename.endswith(".abf"):
        try:
            file_path = os.path.join(root,filename)
            abf = pyabf.ABF(file_path)
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
dv_cut = input("Enter the threshold cut off for the derivative (Allen defaults 20mv/s): ")
try: 
    dv_cut = int(dv_cut)
except:
    dv_cut = 20
tp_cut = input("Enter the threshold cut off for threshold-to-peak (Allen defaults 5ms)[in ms]: ")
try: 
    tp_cut = float(tp_cut/1000)
except:
    tp_cut = 0.005

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
        
def plotabf(abf, spiketimes, lowerlim, upperlim):
    cm = plt.get_cmap("Dark2") #Changes colour based on sweep number
    colors = [cm(x/abf.sweepCount * 0.75) for x in abf.sweepList]
    plt.figure(figsize=(16,6))
    plt.autoscale(True)
    plt.grid(alpha=0)

    plt.xlabel(abf.sweepLabelX)
    plt.ylabel(abf.sweepLabelY)


    for sweepNumber in range(abf.sweepCount - 2, abf.sweepCount):
        abf.setSweep(sweepNumber)
        i1, i2 = int(abf.dataRate * lowerlim), int(abf.dataRate * upperlim) # plot part of the sweep
        dataX = abf.sweepX[i1:i2]
        dataY = abf.sweepY[i1:i2]
        colour = colors[sweepNumber]
        plt.plot(dataX, dataY, color=colour, alpha=1, lw=1)
        

    plt.vlines(spiketimes, 20, 30, linestyles='dashed', zorder=99)
    plt.savefig(abf.abfID +'.png', dpi=600)
    plt.close(fig='all')
debugplot = 0

dfs = pd.DataFrame()
df_spike_count = pd.DataFrame()
for root,dir,fileList in os.walk(files):
 for filename in fileList:
    if filename.endswith(".abf"):
            file_path = os.path.join(root,filename)
        #try:
            abf = pyabf.ABF(file_path)
        
            if abf.sweepLabelY != 'Clamp Current (pA)' and abf.protocol != 'Gap free' and protocol_name in abf.protocol:
                print(filename + ' import')
              #try:
                np.nan_to_num(abf.data, nan=-9999, copy=False)
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
                temp_spike_df = pd.DataFrame()
                temp_spike_df['__a_filename'] = [abf.abfID]
                full_dataI = []
                full_dataV = []
                for sweepNumber in range(0, sweepcount): 
                    real_sweep_length = abf.sweepLengthSec - 0.0001
                    if sweepNumber < 9:
                        real_sweep_number = '00' + str(sweepNumber + 1)
                    elif sweepNumber > 8 and sweepNumber < 99:
                        real_sweep_number = '0' + str(sweepNumber + 1)

                
                    if lowerlim == 0 and upperlim == 0:
                    
                        upperlim = real_sweep_length
                        spikext = feature_extractor.SpikeFeatureExtractor(filter=filter, dv_cutoff=dv_cut, end=upperlim, max_interval=tp_cut)
                        spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=0, end=upperlim)
                    elif upperlim > real_sweep_length:
                    
                        upperlim = real_sweep_length
                        spikext = feature_extractor.SpikeFeatureExtractor(filter=filter, dv_cutoff=dv_cut, start=lowerlim, end=upperlim, max_interval=tp_cut)
                        spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=lowerlim, end=upperlim)
                        #spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=lowerlim, end=upperlim)
                    else:
                        #upperlim = real_sweep_length
                        spikext = feature_extractor.SpikeFeatureExtractor(filter=filter, dv_cutoff=dv_cut, start=lowerlim, end=upperlim, max_interval=tp_cut)

                        spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=lowerlim, end=upperlim)

                    abf.setSweep(sweepNumber)
               
                    dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
                    full_dataI.append(dataI)
                    full_dataV.append(dataV)
                    if dataI.shape[0] < dataV.shape[0]:
                        dataI = np.hstack((dataI, np.full(dataV.shape[0] - dataI.shape[0], 0)))
                
                    try:
               
                        uindex = abf.epochPoints[1] + 1
                    except:
                        uindex = 0
                    spike_in_sweep = spikext.process(dataT, dataV, dataI)
                    spike_train = spiketxt.process(dataT, dataV, dataI, spike_in_sweep)
                    spike_count = spike_in_sweep.shape[0]
                    temp_spike_df["Sweep " + real_sweep_number + " spike count"] = [spike_count]
                    current_str = np.array2string(np.unique(dataI))
                    current_str = current_str.replace('[', '')
                    current_str = current_str.replace(' 0.', '')
                    current_str = current_str.replace(']', '')
                    temp_spike_df["Current_Sweep " + real_sweep_number + " current injection"] = [current_str]
                    #decay_fast, decay_slow = exp_decay_factor(dataT, dataV, dataI, 3000)
                    #temp_spike_df["fast decay" + real_sweep_number] = [decay_fast]
                    #temp_spike_df["slow decay" + real_sweep_number] = [decay_slow]
                    if dataI[np.argmin(dataI)] < 0:
                        try:
                            if lowerlim < 0.1:
                                b_lowerlim = 0.1
                            else:
                                b_lowerlim = lowerlim
                            temp_spike_df['baseline voltage' + real_sweep_number] = subt.baseline_voltage(dataT, dataV, start=b_lowerlim)
                            temp_spike_df['sag' + real_sweep_number] = subt.sag(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                            temp_spike_df['time_constant' + real_sweep_number] = subt.time_constant(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                            
                            #temp_spike_df['voltage_deflection' + real_sweep_number] = subt.voltage_deflection(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                        except:
                            print("Subthreshold Processing Error with " + str(abf.abfID))

                    if spike_count > 0:
                        temp_spike_df["isi_Sweep " + real_sweep_number + " isi"] = [spike_train['first_isi']]
                        
                        spike_train_df = pd.DataFrame(spike_train, index=[0])
                        nan_series = pd.DataFrame(np.full(abs(spike_count-1), np.nan))
                        #spike_train_df = spike_train_df.append(nan_series)
                        spike_in_sweep['spike count'] = np.hstack((spike_count, np.full(abs(spike_count-1), np.nan)))
                        spike_in_sweep['sweep Number'] = np.hstack(((sweepNumber+1), np.full(abs(spike_count-1), np.nan)))
                        temp_spike_df["spike_" + real_sweep_number + " 1"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[0] - spike_in_sweep['threshold_v'].to_numpy()[0])
                        temp_spike_df["spike_" + real_sweep_number + "AHP 1"] = spike_in_sweep['fast_trough_v'].to_numpy()[0]
                        temp_spike_df["spike_" + real_sweep_number + "AHP height 1"] = abs(spike_in_sweep['peak_v'].to_numpy()[0] - spike_in_sweep['fast_trough_v'].to_numpy()[0])
                        temp_spike_df["latency_" + real_sweep_number + "latency"] = spike_train['latency']
                        temp_spike_df["width_spike" + real_sweep_number + "1"] = spike_in_sweep['width'].to_numpy()[0]
                        
                        temp_spike_df["exp growth" + real_sweep_number] = [exp_growth_factor(dataT, dataV, dataI, spike_in_sweep['threshold_index'].to_numpy()[0])]
                        
                        if spike_count > 2:
                            f_isi = spike_in_sweep['peak_t'].to_numpy()[-1]
                            l_isi = spike_in_sweep['peak_t'].to_numpy()[-2]
                            temp_spike_df["last_isi" + real_sweep_number + " isi"] = [abs( f_isi- l_isi )]
                            spike_in_sweep['isi_'] = np.hstack((np.diff(spike_in_sweep['peak_t'].to_numpy()), np.nan))
                            temp_spike_df["min_isi" + real_sweep_number + " isi"] = np.nanmin(np.hstack((np.diff(spike_in_sweep['peak_t'].to_numpy()), np.nan)))
                            temp_spike_df["spike_" + real_sweep_number + " 2"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[1] - spike_in_sweep['threshold_v'].to_numpy()[1])
                            temp_spike_df["spike_" + real_sweep_number + " 3"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[-1] - spike_in_sweep['threshold_v'].to_numpy()[-1])
                            temp_spike_df["spike_" + real_sweep_number + "AHP 2"] = spike_in_sweep['fast_trough_v'].to_numpy()[1]
                            temp_spike_df["spike_" + real_sweep_number + "AHP 3"] = spike_in_sweep['fast_trough_v'].to_numpy()[-1]
                            temp_spike_df["spike_" + real_sweep_number + "AHP height 2"] = abs(spike_in_sweep['peak_v'].to_numpy()[1] - spike_in_sweep['fast_trough_v'].to_numpy()[1])
                            temp_spike_df["spike_" + real_sweep_number + "AHP height 3"] = abs(spike_in_sweep['peak_v'].to_numpy()[-1] - spike_in_sweep['fast_trough_v'].to_numpy()[-1])
                            temp_spike_df["width_spike" + real_sweep_number + "2"] = spike_in_sweep['width'].to_numpy()[1]
                            temp_spike_df["width_spike" + real_sweep_number + "3"] = spike_in_sweep['width'].to_numpy()[-1]
                        else:
                            temp_spike_df["last_isi" + real_sweep_number + " isi"] = [np.nan]
                            spike_in_sweep['isi_'] = np.hstack((np.full(abs(spike_count), np.nan)))
                            temp_spike_df["min_isi" + real_sweep_number + " isi"] = [spike_train['first_isi']]
                        spike_in_sweep = spike_in_sweep.join(spike_train_df)
                        print("Processed Sweep " + str(sweepNumber+1) + " with " + str(spike_count) + " aps")
                        df = df.append(spike_in_sweep, ignore_index=True, sort=True)
                    else:
                        temp_spike_df["latency_" + real_sweep_number + "latency"] = [np.nan]
                        temp_spike_df["isi_Sweep " + real_sweep_number + " isi"] = [np.nan]
                        temp_spike_df["last_isi" + real_sweep_number + " isi"] = [np.nan]
                        temp_spike_df["spike_" + real_sweep_number + " 1"] = [np.nan]
                        temp_spike_df["spike_" + real_sweep_number + " 2"] = [np.nan]
                        temp_spike_df["spike_" + real_sweep_number + " 3"] = [np.nan]
                        temp_spike_df["spike_" + real_sweep_number + "AHP 1"] = [np.nan]
                        temp_spike_df["spike_" + real_sweep_number + "AHP height 1"] = [np.nan]
                        temp_spike_df["spike_" + real_sweep_number + "AHP 2"] = [np.nan]
                        temp_spike_df["spike_" + real_sweep_number + "AHP 3"] = [np.nan]
                        temp_spike_df["spike_" + real_sweep_number + "AHP height 2"] = [np.nan]
                        temp_spike_df["spike_" + real_sweep_number + "AHP height 3"] = [np.nan]
                        temp_spike_df["latency_" + real_sweep_number + "latency"] = [np.nan]
                        temp_spike_df["width_spike" + real_sweep_number + "1"] = [np.nan]
                        temp_spike_df["width_spike" + real_sweep_number + "2"] = [np.nan]
                        temp_spike_df["width_spike" + real_sweep_number + "3"] = [np.nan]
                        temp_spike_df["min_isi" + real_sweep_number + " isi"] = [np.nan]
                        temp_spike_df["exp growth" + real_sweep_number] = [np.nan]
                temp_spike_df['protocol'] = [abf.protocol]
                if df.empty:
                    df = df.assign(__file_name=np.full(1,abf.abfID))
                    print('no spikes found')
                else:
                    df = df.assign(__file_name=np.full(len(df.index),abf.abfID))
                    abf.setSweep(int(df['sweep Number'].to_numpy()[0] - 1))
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
                    temp_spike_df["mean_current"] = [np.mean(df['peak_i'].to_numpy())]
                    temp_spike_df["mean_latency"] = [np.mean(df['latency'].to_numpy())]
                    temp_spike_df["mean_thres"] = [np.mean(df['threshold_v'].to_numpy())]
                    temp_spike_df["mean_width"] = [np.mean(df['width'].to_numpy())]
                    temp_spike_df["mean_heightPT"] = [np.mean(abs(df['peak_v'].to_numpy() - df['fast_trough_v'].to_numpy()))]
                    temp_spike_df["mean_heightTP"] = [np.mean(abs(df['threshold_v'].to_numpy() - df['peak_v'].to_numpy()))]
                    temp_spike_df["mean_upstroke"] = [np.mean(df['upstroke'].to_numpy())]
                    temp_spike_df["mean_downstroke"] = [np.mean(df['upstroke'].to_numpy())]
                    temp_spike_df["mean_fast_trough"] = [np.mean(df['fast_trough_v'].to_numpy())]
                    spiketimes = np.ravel(df['peak_t'].to_numpy())
                    plotabf(abf, spiketimes, lowerlim, upperlim)
                full_dataI = np.vstack(full_dataI) 
                full_dataV = np.vstack(full_dataV) 
                decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p = exp_decay_factor(dataT, np.mean(full_dataV,axis=0), np.mean(full_dataI,axis=0), 3000, abf_id=abf.abfID)
                
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
                cols = df.columns.tolist()
                cols = cols[-1:] + cols[:-1]
                df = df[cols]
                df = df[cols]
            
            
                if bfeatcon == True:
                   df_spike_count = df_spike_count.append(temp_spike_df, sort=True)
                   dfs = dfs.append(df, sort=True)
              #except:
               # print('Issue Processing ' + filename)

            else:
                print('Not correct protocol: ' + abf.protocol)
        #except:
          # print('Issue Processing ' + filename)


try:
 if featfile == True:
    ids = dfs['__file_name'].unique()
    


    tempframe = dfs.groupby('__file_name').mean().reset_index()
    tempframe.to_csv(root_fold + '/allAVG_' + tag + '.csv')

 if featrheo == True:
    tempframe = dfs.drop_duplicates(subset='__file_name')
    tempframe.to_csv(root_fold + '/allRheo_' + tag + '.csv')

        



 if bfeatcon == True:
    df_spike_count.to_csv(root_fold + '/spike_count_' + tag + '.csv')
    dfs.to_csv(root_fold + '/allfeatures_' + tag + '.csv')
except: 
    print('error saving')

print("==== SUCCESS ====")
input('Press ENTER to exit')