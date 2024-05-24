import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf
import copy
import multiprocessing as mp
from ipfx import feature_extractor, spike_detector, time_series_utils
from ipfx import subthresh_features as subt
import scipy.signal as signal
import logging

#Local imports
from .ipfx_df import _build_full_df, _build_sweepwise_dataframe, save_data_frames, save_subthres_data
from .dataset import cellData
from .patch_utils import plotabf, load_protocols, find_non_zero_range, filter_abf
from .patch_subthres import exp_decay_factor, membrane_resistance, mem_cap, mem_cap_alt, \
    rmp_mode, compute_sag, exp_decay_factor_alt, exp_growth_factor, determine_subt, df_select_by_col
from .QC import run_qc

#set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('Feature extractor loaded')

#this is here, to swap functions in the feature extractor for ones specfic to the INOUE lab IC1 standard protocol
IC1_SPECIFIC_FUNCTIONS = True
default_dict = {'start': 0, 'end': 0, 'filter': 0, 'stim_find': True}

def folder_feature_extract(files, param_dict, plot_sweeps=-1, protocol_name='IC1', n_jobs=1):
    """
    Runs the full ipfx / smestern feature extraction pipeline over a folder of files, list of files, or a list of cellData objects.
    Returns a dataframe of the full data as returned by the ipfx feature extractor. Consists of all the sweeps in the files stacked on top of each other.
    Args:
        files (list): _description_
        param_dict (dict): _description_
        plot_sweeps (int, bool, optional): _description_. Defaults to -1.
        protocol_name (str, optional): _description_. Defaults to 'IC1'.

    Returns:
        df_raw_out: A dataframe of the full data as returned by the ipfx feature extractor. Consists of all the sweeps in the files stacked on top of each other.
        df_spike_count: The standard dataframe of the spike count data. As designed at the inoue lab. Each cell will have a row in this dataframe. returns not only
            the spike count, but also subthreshold features and suprathreshold features.
        df_running_avg_count: The running average of the spike count data. This is a sweepwise dataframe, where each row is the running average of several features.
    """
    if isinstance(files, str) or not isinstance(files, list):
        filelist = glob.glob(files + "/**/*.abf", recursive=True)
    elif isinstance(files, list):
        #check if the files are strings or cellData objects
        if isinstance(files[0], str):
            filelist = files
        elif isinstance(files[0], cellData):
            filelist = [x.file for x in files]
    else:
        logger.error('Files must be a list of strings, a string, or a list of cellData objects')
        return None, None, None
    
    #create our output dataframes
    spike_count = []
    df_full = []
    df_running_avg = []
    #run the feature extractor
    if n_jobs > 1: #if we are using multiprocessing
        pool = mp.Pool(processes=n_jobs)
        results = [pool.apply(process_file, args=(file, param_dict, plot_sweeps, protocol_name)) for file in filelist]
        pool.close()
        ##split out the results
        for result in results:
            temp_res = result
            df_full.append(temp_res[1])
            df_running_avg.append(temp_res[2])
            spike_count.append(temp_res[0])
        pool.join()
    #if we are not using multiprocessing
    else:
        for f in filelist:
            temp_df_spike_count, temp_full_df, temp_running_bin = process_file(f, copy.deepcopy(param_dict), plot_sweeps, protocol_name)
            spike_count.append(temp_df_spike_count)
            df_full.append(temp_full_df)
            df_running_avg.append(temp_running_bin)

    #concatenate the dataframes
    df_spike_count = pd.concat(spike_count, sort=True)
    df_raw_out = pd.concat(df_full, sort=True)
    df_running_avg_count = pd.concat(df_running_avg, sort=False)
    return df_raw_out, df_spike_count, df_running_avg_count

def process_file(file_path, param_dict, plot_sweeps, protocol_name):
    """Takes an file and runs the feature extractor on it. Filters the protocol etc.
    Essentially a wrapper for the feature extractor. As when there is an error we dont want to stop the whole program, we just want to skip the abf.
    Args:
        file_path (str, os.path): _description_
        param_dict (dict): _description_
        plot_sweeps (bool): _description_
        protocol_name (str): _description_

    Returns:
        spike_dataframe, spikewise_dataframe, running_bin_data_frame : _description_
    """
    try:
        abf = pyabf.ABF(file_path, loadData=False)           
        if protocol_name in abf.protocol: 
            print(file_path + ' import')
            abf = pyabf.ABF(file_path, loadData=True)  #if its the correct protocol, we will reload the abf
            temp_spike_df, df, temp_running_bin = analyze_abf(abf, sweeplist=None, plot=plot_sweeps, param_dict=param_dict)
            return temp_spike_df, df, temp_running_bin
        else:
            print('Not correct protocol: ' + abf.protocol)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except:
       return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


def analyze_abf(abf, sweeplist=None, plot=-1, param_dict=None):
    """_summary_

    Args:
        abf (_type_): _description_
        sweeplist (_type_, optional): _description_. Defaults to None.
        plot (int, optional): _description_. Defaults to -1.
        param_dict (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    np.nan_to_num(abf.data, nan=-9999, copy=False)

    #load the data 
    x, y ,c = loadABF(abf.abfFilePath)

    #If there is more than one sweep, we need to ensure we dont iterate out of range
    if sweeplist == None:
        if abf.sweepCount > 1:
            sweepcount = abf.sweepList
        else:
            sweepcount = [0]
    
    #Now we walk through the sweeps looking for action potentials
    df = pd.DataFrame()
    temp_spike_df = pd.DataFrame()
    temp_spike_df['filename'] = [abf.abfID]
    temp_spike_df['foldername'] = [os.path.dirname(abf.abfFilePath)]
    temp_running_bin = pd.DataFrame()
    
    
    #for now if user wants to filter by stim time we will just use the first sweep
    stim_find = param_dict.pop('stim_find')
    if stim_find:
        abf.setSweep(abf.sweepList[-1])
        start, end = find_non_zero_range(abf.sweepX, abf.sweepC)
        param_dict['end'] = end
        param_dict['start'] = start
        print('Stimulation time found: ' + str(start) + ' to ' + str(end))

    #if the user wants a bessel filter pop it out of the param_dict
    if 'bessel_filter' in param_dict:
        bessel_filter = param_dict.pop('bessel_filter')
    else:
        bessel_filter = None

    #iterate through the sweeps
    for sweepNumber in sweepcount: 
        real_sweep_length = abf.sweepLengthSec - 0.0001
        abf.setSweep(sweepNumber)
        #here we just make sure the sweep number is in the correct format for the dataframe
        real_sweep_number = sweepNumber_to_real_sweep_number(sweepNumber)
        if param_dict['start'] == 0 and param_dict['end'] == 0: 
            param_dict['end']= real_sweep_length
        elif param_dict['end'] > real_sweep_length:
            param_dict['end'] = real_sweep_length
        
        spike_in_sweep, spike_train = analyze_spike_sweep(abf, sweepNumber, param_dict, bessel_filter=bessel_filter) ### Returns the default Dataframe Returned by ipfx
        
        #build the dataframe, this will be the dataframe that is used for the full data, essentially the sweepwise dataframe, each file will have a dataframe like this
        temp_spike_df, df, temp_running_bin = _build_sweepwise_dataframe(real_sweep_number, spike_in_sweep, spike_train, temp_spike_df, df, temp_running_bin, param_dict) 
        
        #attach the custom features
        custom_features = _custom_sweepwise_features(x[sweepNumber], y[sweepNumber] ,c[sweepNumber] , real_sweep_number, param_dict, temp_spike_df, spike_in_sweep)
        temp_spike_df = temp_spike_df.assign(**custom_features)

    #add the filename and foldername to the temp_running_bin
    temp_running_bin['filename'] = abf.abfID
    temp_running_bin['foldername'] = os.path.dirname(abf.abfFilePath)
    #compute some final features, here we need all the sweeps etc, so these are computed after the sweepwise features
    temp_spike_df = _custom_full_features(x, y, c, param_dict, temp_spike_df)
    temp_spike_df, df, temp_running_bin = _build_full_df(abf, temp_spike_df, df, temp_running_bin, sweepcount)
    
    return temp_spike_df, df, temp_running_bin

def analyze_spike_sweep(x, y, c, sweepNumber, param_dict, bessel_filter=None):
    """_summary_

    Args:
        abf (_type_): _description_
        sweepNumber (_type_): _description_
        param_dict (_type_): _description_
        bessel_filter (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    abf.setSweep(sweepNumber)
    spikext = feature_extractor.SpikeFeatureExtractor(**param_dict)
    spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=param_dict['start'], end=param_dict['end'])  
    dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
    #if the user asks for a filter, apply it
    if bessel_filter is not None:
        if bessel_filter != -1:
            dataV = filter_abf(dataV, abf, bessel_filter)
    if dataI.shape[0] < dataV.shape[0]:
                dataI = np.hstack((dataI, np.full(dataV.shape[0] - dataI.shape[0], 0)))
    spike_in_sweep = spikext.process(dataT, dataV, dataI) #returns the default Dataframe Returned by ipfx
    spike_train = spiketxt.process(dataT, dataV, dataI, spike_in_sweep) #additional dataframe returned by ipfx, contains the features related to consecutive spikes
    return spike_in_sweep, spike_train

def sweepNumber_to_real_sweep_number(sweepNumber):
    if sweepNumber < 9:
            real_sweep_number = '00' + str(sweepNumber + 1)
    elif sweepNumber > 8 and sweepNumber < 99:
            real_sweep_number = '0' + str(sweepNumber + 1)
    return real_sweep_number

#CUSTOM FEATURES
def _custom_sweepwise_features(sweepX, sweepY, sweepC, real_sweep_number, param_dict, spike_df, rawspike_df):
    custom_features = {}

    sag, taum, voltage = subthres_a(sweepX, sweepY, sweepC, param_dict['start'], param_dict['end'])
    custom_features["Sag Ratio " + real_sweep_number + ""] = sag
    custom_features["Taum " + real_sweep_number + ""] = taum

    try:
        custom_features['baseline voltage' + real_sweep_number] = subt.baseline_voltage(sweepX, sweepY, start=0.1, filter_frequency=param_dict['filter'])
    except:
        print('Fail to find baseline voltage')
    
    #compute features if there was a spike
    #if spike_df.empty:
        #custom_features["exp growth tau1 " + real_sweep_number] = np.nan
        #custom_features["exp growth tau2 " + real_sweep_number] = np.nan
    #else:
        #pass
        #end_index = rawspike_df['threshold_t'][0] if 'threshold_t' in rawspike_df.columns else 0.7
        #curve = exp_growth_factor(sweepX, sweepY, sweepC, alpha=1/taum, end_index=end_index )
        #custom_features["exp growth tau1 " + real_sweep_number] = curve[2]
        #custom_features["exp growth tau2 " + real_sweep_number] = curve[-1]
    return custom_features


def _custom_full_features(x,y,c, param_dict, spike_df):
    #gather some protocol information that is requested by patchers
    #some more advanced current injection features
    spike_df = merge_current_injection_features(x, y, c, spike_df)
    #try qc or just return the dataframe
    try:
        _qc_data = run_qc(y, c)
        spike_df['QC Mean RMS'] = _qc_data[0]
        spike_df['QC Mean Sweep Drift'] = _qc_data[2]
    except:
        spike_df['QC Mean RMS'] = np.nan
        spike_df['QC Mean Sweep Drift'] = np.nan
    return spike_df


def _merge_current_injection_features(sweepX, sweepY, sweepC, spike_df):
    """
    This function will compute the current injection features for a given sweep. It will return a dictionary of the features, which will be appended to the dataframe.
    Tries to capture the current injection features of the sweep, such as the current at each epoch, the delta between epochs, the stimuli length, and the sample rates.
    takes:
        sweepX (np.array): The time array of the sweep
        sweepY (np.array): The voltage array of the sweep
        sweepC (np.array): The current array of the sweep
        spike_df (pd.DataFrame): The dataframe that will be appended with the new features
    returns:
        spike_df (pd.DataFrame): The dataframe that has been appended with the new features
    """
    new_current_injection_features = {}
    #first we want to compute the number of epochs, take the diff along the columns and count the number of nonzero rows
    diffC = np.diff(sweepC, axis=1)
    #find the row with the most nonzero entries
    most_nonzero = np.argmax(np.count_nonzero(diffC, axis=1))
    idx_epochs = np.sort(np.unique(diffC[most_nonzero], return_index=True)[1]) +1
    #now iter the sweeps and find the current at each epoch
    for j, idx in enumerate(idx_epochs):
        currents = sweepC[:, idx]
        if np.any(currents): #if its not zero
            if len(np.unique(currents)) == 1:
                new_current_injection_features[f"current_sweep_all_epoch_{j}"] = currents[0]
            else:
                new_current_injection_features.update({f"current_sweep_{sweepNumber_to_real_sweep_number(i)}_epoch_{j}": current for i, current in enumerate(currents)})
                new_current_injection_features[f"current_epoch_{j}_delta"] = np.diff(currents)[0]
     
    #finally we want to compute the stimuli size
    #compute the dt
    dt = np.diff(sweepX[0])[0]
    #first compute the sweepwise variation by taking the diff along the columns
    #compute the length of the nonzero current injections:
    stimuli_length = [len(np.flatnonzero(row))*dt for row in sweepC if np.any(row)]
    #pack it into the dict, if there is only one stimuli length, we will just take that
    if len(np.unique(stimuli_length)) == 1:
        new_current_injection_features['stimuli_length'] = stimuli_length[0]
    else:
        #take the mode of the stimuli length
        new_current_injection_features['stimuli_length'] = np.unique(stimuli_length)[np.unique(stimuli_length, return_counts=True)[1].argmax()]
       # for i, stimuli in enumerate(stimuli_length):
         #   new_current_injection_features[f"stimuli_length_sweep_{i}"] = stimuli

    #also compute the sample_rate
    new_current_injection_features['sample_rate'] = np.round(1/dt, 2)

    #ideally we would not modify the dataframe in place, but this is the easiest way to do it
    spike_df = spike_df.assign(**new_current_injection_features)
    return spike_df


def _merge_current_injection_features_IC1(sweepX, sweepY, sweepC, spike_df):
    """
    This function will compute the current injection features for a given sweep. It will return a dictionary of the features, which will be appended to the dataframe.
    THIS FUNCTION IS SPECIFIC TO THE INOUE LAB IC1 PROTOCOL. Which consists of a hyperpolarizing current injection followed by a depolarizing current injection.
    Tries to capture the current injection features of the sweep, such as the current at each epoch, the delta between epochs, the stimuli length, and the sample rates.
    takes:
        sweepX (np.array): The time array of the sweep
        sweepY (np.array): The voltage array of the sweep
        sweepC (np.array): The current array of the sweep
        spike_df (pd.DataFrame): The dataframe that will be appended with the new features
    returns:
        spike_df (pd.DataFrame): The dataframe that has been appended with the new features
    """

    new_current_injection_features = {}
    #first we want to compute the number of epochs, take the diff along the columns and count the number of nonzero rows
    diffC = np.diff(sweepC, axis=1)
    #find the row with the most nonzero entries
    most_nonzero = np.argmax(np.count_nonzero(diffC, axis=1))
    idx_epochs = np.sort(np.unique(diffC[most_nonzero], return_index=True)[1]) +1
    #now iter the sweeps and find the current at each epoch
    non_zero_epochs = idx_epochs[np.any(sweepC[:,idx_epochs ], axis=0)]
    #should be 2 epochs for IC1
    if len(non_zero_epochs) > 2 or len(idx_epochs) > 3:
        print("Warning, more than 2 epochs found in IC1 protocol, taking the first two")
        new_current_injection_features["IC1_protocol_check"] = "Other"
        non_zero_epochs = non_zero_epochs[:2]
    elif len(non_zero_epochs) < 2:
        #if there are less than 2 epochs, we will flag the cell, return no features
        print("Warning, less than 2 epochs found in IC1 protocol, no current injection features will be computed")
        new_current_injection_features["IC1_protocol_check"] = "Other"
        #ideally we would not modify the dataframe in place, but this is the easiest way to do it
        spike_df = spike_df.assign(**new_current_injection_features)
        return spike_df
    

    #get the hyperpolarizing current, which should be the first epoch
    hyperpolarizing_current = sweepC[:, non_zero_epochs[0]]
    if len(np.unique(hyperpolarizing_current)) == 1 & np.all(hyperpolarizing_current==-20):
        new_current_injection_features["hyperpolarizing_current"] = hyperpolarizing_current[0]
    else:
        print("Warning, more than one hyperpolarizing current found in IC1 protocol or different step size, taking the first")
        new_current_injection_features["hyperpolarizing_current"] = hyperpolarizing_current[0]
        if "IC1_protocol_check" not in new_current_injection_features:
            new_current_injection_features["IC1_protocol_check"] = "IC1-Modified-step"
    
    #get the depolarizing current, which should be the second epoch
    depolarizing_current = sweepC[:, non_zero_epochs[1]]
    #get the delta between the two currents
    if sweepC.shape[0] < 2:
        new_current_injection_features['depolarizing_current_delta'] = np.nan
    else:
        new_current_injection_features['depolarizing_current_delta'] = depolarizing_current[1] - depolarizing_current[0]
    new_current_injection_features.update({f"depolarizing_current_sweep_{sweepNumber_to_real_sweep_number(i)}": current for i, current in enumerate(depolarizing_current)})
    #if the delta is not 10 or consistent across sweeps, we will will flag the cell
    if np.all(np.diff(depolarizing_current)!=10) or new_current_injection_features['depolarizing_current_delta']!=10:
        if "IC1_protocol_check" not in new_current_injection_features:
            new_current_injection_features["IC1_protocol_check"] = "IC1-Modified-step"


    #finally we want to compute the stimuli size
    #compute the dt
    dt = np.diff(sweepX[0])[0]
    #use the last sweep to compute the stimuli length
    hyperpolarizing_stimuli_length = np.round(len(sweepC[-1, (sweepC[-1]<0.0)])*dt, 4)
    depolarizing_stimuli_length = np.round(len(sweepC[-1, (sweepC[-1]>0.0)])*dt, 4)

    new_current_injection_features['hyperpolarizing_stimuli_length'] = hyperpolarizing_stimuli_length
    new_current_injection_features['depolarizing_stimuli_length'] = depolarizing_stimuli_length

    if hyperpolarizing_stimuli_length != 0.3 or depolarizing_stimuli_length != 0.7:
        if "IC1_protocol_check" not in new_current_injection_features:
            new_current_injection_features["IC1_protocol_check"] = "IC1-Modified-length"
    
    if "IC1_protocol_check" not in new_current_injection_features:
        new_current_injection_features["IC1_protocol_check"] = "IC1"

    #also compute the sample_rate
    new_current_injection_features['sample_rate'] = np.round(1/dt, 2)

    #ideally we would not modify the dataframe in place, but this is the easiest way to do it
    spike_df = spike_df.assign(**new_current_injection_features)
    return spike_df

if IC1_SPECIFIC_FUNCTIONS:
    merge_current_injection_features = _merge_current_injection_features_IC1
else:
    merge_current_injection_features = _merge_current_injection_features


#SUBTHRESHOLD FEATURES
def preprocess_abf_subthreshold(file_path, protocol_name='', param_dict={}):
    #try:
    abf = pyabf.ABF(file_path, loadData=False)           
    if protocol_name in abf.protocol:
        print(file_path + ' import')
        abf = pyabf.ABF(file_path)      
        df, avg = analyze_subthres(abf, **param_dict)
        return df, avg
    else:
        print('Not correct protocol: ' + abf.protocol)
        return pd.DataFrame(), pd.DataFrame()
    #except:
       #return pd.DataFrame(), pd.DataFrame()

def analyze_subthres(abf, protocol_name='', savfilter=0, start_sear=None, end_sear=None, subt_sweeps=None, time_after=50, bplot=False):
    filename = abf.abfID
    
    print(filename + ' import')
    file_path = abf.abfID
    root_fold = os.path.dirname(file_path)
    np.nan_to_num(abf.data, nan=-9999, copy=False)
    if savfilter >0:
        abf.data = signal.savgol_filter(abf.data, savfilter, polyorder=3)
    
    #determine the search area
    dataT = abf.sweepX #sweeps will need to be the same length
    if start_sear != None:
        idx_start = np.argmin(np.abs(dataT - start_sear))
    else:
        idx_start = 0
        
    if end_sear != None:
        idx_end = np.argmin(np.abs(dataT - end_sear))
        
    else:
        idx_end = -1



    #If there is more than one sweep, we need to ensure we dont iterate out of range
    if abf.sweepCount > 1:
        if subt_sweeps is None:
            sweepList = determine_subt(abf, (idx_start, idx_end))
            sweepcount = len(sweepList)
        else:
            subt_sweeps_temp = subt_sweeps - 1
            sweep_union = np.intersect1d(abf.sweepList, subt_sweeps_temp)
            sweepList = sweep_union
            sweepcount = 1
    else:
        sweepcount = 1
        sweepList = [0]
    
    
    temp_df = pd.DataFrame()
    temp_df['filename'] = [abf.abfID]
    temp_df['foldername'] = [os.path.dirname(file_path)]
    temp_avg = pd.DataFrame()
    temp_avg['filename'] = [abf.abfID]
    temp_avg['foldername'] = [os.path.dirname(file_path)]
    
    full_dataI = []
    full_dataV = []
    for sweepNumber in sweepList: 
        real_sweep_length = abf.sweepLengthSec - 0.0001
        if sweepNumber < 9:
            real_sweep_number = '00' + str(sweepNumber + 1)
        elif sweepNumber > 8 and sweepNumber < 99:
            real_sweep_number = '0' + str(sweepNumber + 1)

        

        abf.setSweep(sweepNumber)

        dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
        dataT, dataV, dataI = dataT[idx_start:idx_end], dataV[idx_start:idx_end], dataI[idx_start:idx_end]
        dataT = dataT - dataT[0]
        

        decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, p_decay = exp_decay_factor(dataT, dataV, dataI, time_after, abf_id=abf.abfID)
        
        resist = membrane_resistance(dataT, dataV, dataI)
        Cm2, Cm1 = mem_cap(resist, decay_slow)
        Cm3 = mem_cap_alt(resist, decay_slow, curve[3], np.amin(dataI))
        temp_df[f"_1 phase decay {real_sweep_number}"] = [p_decay]           
        temp_df[f"fast 2 phase decay {real_sweep_number}"] = [decay_fast]
        temp_df[f"slow 2 phase decay {real_sweep_number}"] = [decay_slow]
        temp_df[f"Curve fit A {real_sweep_number}"] = [curve[0]]
        temp_df[f"Curve fit b1 {real_sweep_number}"] = [curve[1]]
        temp_df[f"Curve fit b2 {real_sweep_number}"] = [curve[3]]
        temp_df[f"R squared 2 phase {real_sweep_number}"] = [r_squared_2p]
        temp_df[f"R squared 1 phase {real_sweep_number}"] = [r_squared_1p]
        temp_df[f"RMP {real_sweep_number}"] = [rmp_mode(dataV, dataI)]
        temp_df[f"Membrane Resist {real_sweep_number}"] =  resist / 1000000000 #to gigaohms
        temp_df[f"_2 phase Cm {real_sweep_number}"] =  Cm2 * 1000000000000#to pf farad
        temp_df[f"_ALT_2 phase Cm {real_sweep_number}"] =  Cm3 * 1000000000000
        temp_df[f"_1 phase Cm {real_sweep_number}"] =  Cm1 * 1000000000000
        temp_df[f"Voltage sag {real_sweep_number}"],temp_df[f"Voltage min {real_sweep_number}"] = compute_sag(dataT,dataV,dataI, time_after, plot=bplot, clear=False)
         
        #temp_spike_df['baseline voltage' + real_sweep_number] = subt.baseline_voltage(dataT, dataV, start=b_lowerlim)
        #
        #temp_spike_df['time_constant' + real_sweep_number] = subt.time_constant(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
        #temp_spike_df['voltage_deflection' + real_sweep_number] = subt.voltage_deflection(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)

        

        full_dataI.append(dataI)
        full_dataV.append(dataV)
        if dataI.shape[0] < dataV.shape[0]:
            dataI = np.hstack((dataI, np.full(dataV.shape[0] - dataI.shape[0], 0)))
    
    if bplot == True:
            plt.title(abf.abfID)
            plt.ylim(top=-40)
            plt.xlim(right=0.6)
            #plt.legend()
            plt.savefig(root_fold+'//cm_plots//sagfit'+abf.abfID+'sweep'+real_sweep_number+'.png')
    
    full_dataI = np.vstack(full_dataI) 
    indices_of_same = np.arange(full_dataI.shape[0])
    full_dataV = np.vstack(full_dataV)
    if bplot == True:
        if not os.path.exists(root_fold+'//cm_plots//'):
                os.mkdir(root_fold+'//cm_plots//')   
    print("Fitting Decay")
    decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, p_decay = exp_decay_factor_alt(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0), time_after, abf_id=abf.abfID, plot=bplot, root_fold=root_fold)
    print("Computing Sag")
    grow = exp_growth_factor(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0), 1/decay_slow)
    temp_avg[f"Voltage sag mean"], temp_avg["Voltage Min point"] = compute_sag(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0), time_after, plot=bplot)
    temp_avg[f"Sweepwise Voltage sag mean"], temp_avg["Sweepwise Voltage Min point"] = np.nanmean(df_select_by_col(temp_df, ['Voltage sag'])), np.nanmean(df_select_by_col(temp_df, ['Voltage min']))
    
    temp_avg["Averaged 1 phase decay "] = [p_decay]           
    temp_avg["Averaged 2 phase fast decay "] = [decay_fast]
    temp_avg["Averaged 2 phase slow decay "] = [decay_slow]
    temp_avg["Averaged Curve fit A"] = [curve[0]]
    temp_avg["Averaged Curve fit b1"] = [curve[1]]
    temp_avg["Averaged Curve fit b2"] = [curve[3]]
    temp_avg["Averaged R squared 2 phase"] = [r_squared_2p]
    temp_avg["Averaged R squared 1 phase"] = [r_squared_1p]
    temp_avg[f"Averaged RMP"] = [rmp_mode(np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0))]
    temp_avg["SweepCount Measured"] = [sweepcount]
    temp_avg["Averaged alpha tau"] = [grow[1]]
    temp_avg["Averaged b tau"] = [grow[3]]
    if r_squared_2p > r_squared_1p:
        temp_avg["Averaged Best Fit"] = [2]
    else:
        temp_avg["AverageD Best Fit"] = [1]
    print(f"fitting Membrane resist")
    resist = membrane_resistance(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0))
    resist_alt = exp_rm_factor(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0), time_after, decay_slow, abf_id=abf.abfID,  root_fold=root_fold)
    Cm2, Cm1 = mem_cap(resist, decay_slow, p_decay)
    Cm3 = mem_cap_alt(resist, decay_slow, curve[3], np.amin(np.nanmean(full_dataI[indices_of_same,:],axis=0)))
    rm_alt = mem_resist_alt(Cm3, decay_slow)
    temp_avg["Averaged Membrane Resist"] =  resist  / 1000000000 #to gigaohms
    temp_avg["Averaged Membrane Resist _ ALT"] =  resist_alt[0]  / 1000000000
    temp_avg["Averaged Membrane Resist _ ALT 2"] =  rm_alt  / 1000000000#to gigaohms
    temp_avg["Averaged pipette Resist _ ALT"] =  resist_alt[2]  / 1000000000 #to gigaohms
    temp_avg["Averaged 2 phase Cm"] =  Cm2 * 1000000000000
    temp_avg["Averaged 2 phase Cm Alt"] =  Cm3 * 1000000000000
    temp_avg["Averaged 1 phase Cm"] =  Cm1 * 1000000000000
    print(f"Computed a membrane resistance of {(resist  / 1000000000)} giga ohms, and a capatiance of {Cm2 * 1000000000000} pF, and tau of {decay_slow*1000} ms")
    return temp_df, temp_avg
    


if __name__ == '__main__':


    mp.freeze_support()




# def compute_sweepwise_current_injection_features(sweepC, real_sweep_number):
#     current_injection_features = {}
#     #
#     #get the unique current injections, only nonzero
#     unique_current_injections = np.unique(sweepC[np.flatnonzero(sweepC)])
#     #figure out how many current injections there are, if there are more than 6, we will only take the first 3 and last 3
#     if len(unique_current_injections) > 6:
#         unique_current_injections = np.hstack((unique_current_injections[:3], unique_current_injections[-3:]))
#     #append them in the order they appear in the sweepC array
#     unique_negative_current_injections = unique_current_injections[unique_current_injections<0]
#     unique_positive_current_injections = unique_current_injections[unique_current_injections>0]
#     if len(unique_negative_current_injections) > 0:
#         for i, current_injection in enumerate(unique_negative_current_injections[:1]): #here we are only using the first hyperpolarizing current injection
#             #THIS IS basically a special case for INOUE lab standard. However, we are neglecting information about the other hyperpolarizing current injections
#             current_injection_features[f"sweep_{real_sweep_number}_hyperpolarizing_{str(i)}_current"] = current_injection
#          #if there are current injections that are non-positive and have not been accounted for
#         #stack the remaining current injections
#         if len(unique_negative_current_injections) > 1:
#             unique_positive_current_injections = np.hstack((unique_positive_current_injections, unique_negative_current_injections[1:]))

#     if len(unique_positive_current_injections) > 0:
#         for i, current_injection in enumerate(unique_positive_current_injections):
#             current_injection_features[f"sweep_{real_sweep_number}_{str(i)}_current"] = current_injection
#     return current_injection_features
