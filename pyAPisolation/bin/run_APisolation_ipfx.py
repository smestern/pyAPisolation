
print("Loading...")
import sys
import numpy as np
from numpy import genfromtxt
import pyabf
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
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
        


debugplot = 0

dfs = pd.DataFrame()
df_spike_count = pd.DataFrame()
for root,dir,fileList in os.walk(files):
 for filename in fileList:
    if filename.endswith(".abf"):
     file_path = os.path.join(root,filename)
     try:
        abf = pyabf.ABF(file_path)
        
        if abf.sweepLabelY != 'Clamp Current (pA)' and abf.protocol != 'Gap free' and protocol_name in abf.protocol:
          print(filename + ' import')
          try:
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
            for sweepNumber in range(0, sweepcount): 
                real_sweep_length = abf.sweepLengthSec - 0.0001
                if sweepNumber < 9:
                    real_sweep_number = '00' + str(sweepNumber + 1)
                elif sweepNumber > 8 and sweepNumber < 99:
                    real_sweep_number = '0' + str(sweepNumber + 1)

                
                if lowerlim == 0 and upperlim == 0:
                    
                    upperlim = real_sweep_length
                    spikext = feature_extractor.SpikeFeatureExtractor(filter=filter, dv_cutoff=dv_cut, end=upperlim)
                    spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=0, end=upperlim)
                elif upperlim > real_sweep_length:
                    
                    upperlim = real_sweep_length
                    spikext = feature_extractor.SpikeFeatureExtractor(filter=filter, dv_cutoff=dv_cut, start=lowerlim, end=upperlim)
                    spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=lowerlim, end=upperlim)
                    #spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=lowerlim, end=upperlim)
                else:
                    #upperlim = real_sweep_length
                    spikext = feature_extractor.SpikeFeatureExtractor(filter=filter, dv_cutoff=dv_cut, start=lowerlim, end=upperlim)

                    spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=lowerlim, end=upperlim)

                abf.setSweep(sweepNumber)
               
                dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
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
                if dataI[np.argmin(dataI)] < 0:
                    try:
                        if lowerlim < 0.1:
                            b_lowerlim = 0.1
                        else:
                            b_lowerlim = lowerlim
                        temp_spike_df['baseline voltage' + real_sweep_number] = subt.baseline_voltage(dataT, dataV, start=b_lowerlim)
                        temp_spike_df['sag' + real_sweep_number] = subt.sag(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                        temp_spike_df['time_constant' + real_sweep_number] = subt.time_constant(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                        #temp_spike_df['voltage_deflection' + str(sweepNumber +1)] = subt.voltage_deflection(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                    except:
                        print("Subthreshold Processing Error with " + str(abf.abfID))

                if spike_count > 0:
                    temp_spike_df["isi_Sweep " + real_sweep_number + " isi"] = [spike_train['first_isi']]
                    spike_train_df = pd.DataFrame(spike_train, index=[0])
                    nan_series = pd.DataFrame(np.full(abs(spike_count-1), np.nan))
                    #spike_train_df = spike_train_df.append(nan_series)
                    spike_in_sweep['spike count'] = np.hstack((spike_count, np.full(abs(spike_count-1), np.nan)))
                    spike_in_sweep['sweep Number'] = np.hstack(((sweepNumber+1), np.full(abs(spike_count-1), np.nan)))
                    
                    if spike_count > 2:
                        f_isi = spike_in_sweep['peak_t'].to_numpy()[-1]
                        l_isi = spike_in_sweep['peak_t'].to_numpy()[-2]
                        temp_spike_df["last_isi" + real_sweep_number + " isi"] = [abs( f_isi- l_isi )]
                        spike_in_sweep['isi_'] = np.hstack((np.diff(spike_in_sweep['peak_t'].to_numpy()), np.nan))
                    else:
                        temp_spike_df["last_isi" + real_sweep_number + " isi"] = [np.nan]
                        spike_in_sweep['isi_'] = np.hstack((np.full(abs(spike_count), np.nan)))
                    spike_in_sweep = spike_in_sweep.join(spike_train_df)
                    print("Processed Sweep " + str(sweepNumber+1) + " with " + str(spike_count) + " aps")
                    df = df.append(spike_in_sweep, ignore_index=True, sort=True)
                else:
                    temp_spike_df["isi_Sweep " + real_sweep_number + " isi"] = [np.nan]
                    temp_spike_df["last_isi" + real_sweep_number + " isi"] = [np.nan]
            
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
                
            cols = df.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df = df[cols]
            df = df[cols]
            
            if bfeatcon == True:
               df_spike_count = df_spike_count.append(temp_spike_df, sort=True)
               dfs = dfs.append(df, sort=True)
          except:
            print('Issue Processing ' + filename)

        else:
            print('Not correct protocol: ' + abf.protocol)
     except:
         print('Issue Processing ' + filename)
   

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