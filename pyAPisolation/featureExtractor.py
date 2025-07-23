import glob
import os
import functools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf
import copy
import multiprocessing as mp
import ipfx.spike_detector
from ipfx import feature_extractor
from ipfx import subthresh_features as subt
import scipy.signal as signal
import logging

#Local imports
from .ipfx_df import _build_full_df, _build_sweepwise_dataframe, save_data_frames, save_subthres_data
from .loadFile import loadFile, loadABF
from .dataset import cellData
from .patch_utils import plotabf, load_protocols, find_non_zero_range, filter_bessel, parse_user_input, sweepNumber_to_real_sweep_number
from .patch_subthres import exp_decay_factor, membrane_resistance, mem_cap, mem_cap_alt, \
    rmp_mode, compute_sag, exp_decay_factor_alt, exp_growth_factor, determine_subt, df_select_by_col, subthres_a, exp_rm_factor, ladder_rm, \
    mem_resist_alt
    
from .QC import run_qc

#set up the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info('Feature extractor loaded')

#this is here, to swap functions in the feature extractor for ones specfic to the INOUE lab IC1 standard protocol
IC1_SPECIFIC_FUNCTIONS = True
DEFAULT_DICT = {'start': 0, 'end': 0, 'filter': 0, 'stim_find': True}

#=== functional interface for programmatic use ===
def analyze(x=None, y=None, c=None, file=None, param_dict=DEFAULT_DICT, return_summary_frames=False):
    """ Runs the ipfx feature extractor over a single sweep, set of sweeps, or file. Returns the standard ipfx dataframe, and summary dataframes (if requested).
    Args:
        x (np.array, optional): The time array of the sweep. Defaults to None.
        y (np.array, optional): The voltage array of the sweep. Defaults to None.
        c (np.array, optional): The current array of the sweep. Defaults to None.
        file (str, optional): The file path of the sweep. Defaults to None.
        param_dict (dict, optional): The dictionary of parameters that will be passed to the feature extractor. Defaults to None.
        return_summary_frames (bool, optional): If True, will return the summary dataframes. Defaults to False.
    Returns:
        df_raw_out: A dataframe of the full data as returned by the ipfx feature extractor. Consists of all the sweeps in the files stacked on top of each other.
        (optional) df_spike_count (pd.DataFrame): The dataframe that contains the standard ipfx features for the sweep, oriented sweepwise
        (optional) df_running_avg_count (pd.DataFrame): The dataframe that contains the standard ipfx features for the consecutive spikes in the sweep
    """
    data = parse_user_input(x, y, c, file)
    #determine what we should return
    temp_spike_df, df, temp_running_bin = analyze_sweepset(data, sweeplist=None, param_dict=param_dict)
    if return_summary_frames:
        return temp_spike_df, df, temp_running_bin
    return temp_spike_df

#cache the function
def analyze_sweep(x=None, y=None, c=None, param_dict=DEFAULT_DICT, bessel_filter=None):
    """ This function will run the ipfx feature extractor on a single sweep. It will return the spike_in_sweep and spike_train dataframes as returned by the ipfx feature extractor.
    takes:
        x (np.array): The time array of the sweep (1d array)
        y (np.array): The voltage array of the sweep (1d array)
        c (np.array): The current array of the sweep (1d array)
        param_dict (dict): The dictionary of parameters that will be passed to the feature extractor. defaults to the default_dict
        bessel_filter (int): The cutoff frequency of the bessel filter. If -1, no filter will be applied. Defaults to None.
    returns:
        spike_in_sweep (pd.DataFrame): The dataframe that contains the standard ipfx features for the sweep
        spike_train (pd.DataFrame): The dataframe that contains the standard ipfx features for the consecutive spikes in the sweep
    """ 
    spikext = feature_extractor.SpikeFeatureExtractor(**param_dict)
    spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=param_dict['start'], end=param_dict['end'])  
    #if the user asks for a filter, apply it
    if bessel_filter is not None:
        if bessel_filter != -1:
            y = filter_bessel(y, 1/10000, bessel_filter)
    if c.shape[0] < y.shape[0]:
                c = np.hstack((c, np.full(y.shape[0] - c.shape[0], 0)))
    spike_in_sweep = spikext.process(x, y, c) #returns the default Dataframe Returned by ipfx
    spike_train = spiketxt.process(x, y, c, spike_in_sweep) #additional dataframe returned by ipfx, contains the features related to consecutive spikes
    return spike_in_sweep, spike_train

def analyze_sweepset(x=None, y=None, c=None, file=None, sweeplist=None, param_dict=DEFAULT_DICT):
    """ Runs the ifpx feature extractor over a set of sweeps. Returns the standard ipfx dataframe, and summary dataframes.
    Args:
        file (_type_): _description_
        sweeplist (_type_, optional): _description_. Defaults to None.
        plot (int, optional): _description_. Defaults to -1.
        param_dict (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """    
    data = parse_user_input(x, y, c, file)

    #load the data 
    x, y ,c = data.dataX, data.dataY, data.dataC

    #If there is more than one sweep, we need to ensure we dont iterate out of range
    if sweeplist == None:
        if data.sweepCount > 1:
            sweepcount = data.sweepList
        else:
            sweepcount = [0]
    
    #Now we walk through the sweeps looking for action potentials
    df = pd.DataFrame()
    temp_spike_df = pd.DataFrame()
    temp_spike_df['filename'] = [data.name]
    temp_spike_df['foldername'] = [os.path.dirname(data.filePath)]
    temp_running_bin = pd.DataFrame()
    
    
    #memory copy the param_dict, as we will be popping values out of it
    param_dict = copy.deepcopy(param_dict)
    #for now if user wants to filter by stim time we will just use the first sweep
    stim_find = param_dict.pop('stim_find')
    #if the user wants a bessel filter pop it out of the param_dict
    if 'bessel_filter' in param_dict:
        bessel_filter = param_dict.pop('bessel_filter')
    else:
        bessel_filter = None


    if stim_find:
        data.setSweep(data.sweepList[-1])
        start, end = find_non_zero_range(data.sweepX, data.sweepC)
        param_dict['end'] = end
        param_dict['start'] = start
        print('Stimulation time found: ' + str(start) + ' to ' + str(end))



    #iterate through the sweeps
    for sweepNumber in sweepcount: 
        real_sweep_length = data.sweepLengthSec - 0.0001
        data.setSweep(sweepNumber)
        #here we just make sure the sweep number is in the correct format for the dataframe
        real_sweep_number = sweepNumber_to_real_sweep_number(sweepNumber)
        if param_dict['start'] == 0 and param_dict['end'] == 0: 
            param_dict['end']= real_sweep_length
        elif param_dict['end'] > real_sweep_length:
            param_dict['end'] = real_sweep_length
        
        spike_in_sweep, spike_train = analyze_sweep(x[sweepNumber], y[sweepNumber] ,c[sweepNumber], param_dict, bessel_filter=bessel_filter) ### Returns the default Dataframe Returned by ipfx
        
        #build the dataframe, this will be the dataframe that is used for the full data, essentially the sweepwise dataframe, each file will have a dataframe like this
        temp_spike_df, df, temp_running_bin = _build_sweepwise_dataframe(real_sweep_number, spike_in_sweep, spike_train, temp_spike_df, df, temp_running_bin, param_dict) 
        
        #attach the custom features
        custom_features = _custom_sweepwise_features(x[sweepNumber], y[sweepNumber] ,c[sweepNumber] , real_sweep_number, param_dict, temp_spike_df, spike_in_sweep)
        temp_spike_df = temp_spike_df.assign(**custom_features)

    #add the filename and foldername to the temp_running_bin
    temp_running_bin['filename'] = data.name
    temp_running_bin['foldername'] = os.path.dirname(data.filePath)
    #compute some final features, here we need all the sweeps etc, so these are computed after the sweepwise features
    temp_spike_df = _custom_full_features(x, y, c, param_dict, temp_spike_df)
    temp_spike_df, df, temp_running_bin = _build_full_df(data, temp_spike_df, df, temp_running_bin, sweepcount)
    
    return temp_spike_df, df, temp_running_bin


def batch_feature_extract(files, param_dict=None, protocol_name='IC1', n_jobs=1):
    """
    Runs the full ipfx feature extraction pipeline over a folder of files, list of files, or a list of cellData objects.
    Returns a dataframe of the full data as returned by the ipfx feature extractor. Consists of all the sweeps in the files stacked on top of each other.
    Args:
        files (list, str, cellData, list of arrays): The list of files, the folder of files, or the list of cellData objects to be analyzed.
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
        elif isinstance(files[0], pyabf.ABF):
            filelist = [x.name for x in files]
        elif isinstance(files[0], np.ndarray):
            filelist = files
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
        results = [pool.apply(process_file, args=(file, param_dict, protocol_name)) for file in filelist]
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
            temp_df_spike_count, temp_full_df, temp_running_bin = process_file(f, copy.deepcopy(param_dict), protocol_name)
            spike_count.append(temp_df_spike_count)
            df_full.append(temp_full_df)
            df_running_avg.append(temp_running_bin)

    #concatenate the dataframes
    df_spike_count = pd.concat(spike_count, sort=True)
    df_raw_out = pd.concat(df_full, sort=True)
    df_running_avg_count = pd.concat(df_running_avg, sort=False)
    return df_raw_out, df_spike_count, df_running_avg_count



#programmatic functions to retrieve certain dataframes
#e.g. if we only need the spike_times dataframe
subset_frames = {'spike_times': ['peak_t'], 'spike_times_isi': ['peak_t', 'isi'], 'spike_times_isi_sweepwise': ['peak_t', 'threshold_t', 'isi']}

analysis_temp_doc_string = """ This function will run the ipfx feature extractor on a single sweep or set of sweeps. And return /%s/ information.
                            takes:
                                x (np.array): The time array of the sweep (1d or 2d array)
                                y (np.array): The voltage array of the sweep (1d or 2d  array)
                                c (np.array): The current array of the sweep (1d or 2d  array)
                                file (str): The file path of the sweep
                                param_dict (dict): The dictionary of parameters that will be passed to the feature extractor. defaults to the default_dict
                                feature_keys (list): The list of features that we want to extract. Defaults to ['']
                                return_array (bool): If True, will return the dataframe as a numpy array. Defaults to True.
                            returns:
                                df_raw (pd.DataFrame): The dataframe that contains the standard ipfx features for the sweep
                            """


def analyze_template(x=None, y=None, c=None, file=None, param_dict=DEFAULT_DICT, feature_keys=[''], return_array=True):
    """ This function will run the ipfx feature extractor on a single sweep or set of sweeps. And return specific dataframes based on the feature_keys.
    useful for when we only need a subset of the features, eg. spike_times, spike_times_isi, spike_times_isi_sweepwise
    takes:
        x (np.array): The time array of the sweep (1d or 2d array)
        y (np.array): The voltage array of the sweep (1d or 2d  array)
        c (np.array): The current array of the sweep (1d or 2d  array)
        file (str): The file path of the sweep
        param_dict (dict): The dictionary of parameters that will be passed to the feature extractor. defaults to the default_dict
        feature_keys (list): The list of features that we want to extract. Defaults to ['']
        return_array (bool): If True, will return the dataframe as a numpy array. Defaults to True.
    returns:
        df_raw (pd.DataFrame): The dataframe that contains the standard ipfx features for the sweep
    """

    spike_count_df, df_raw, running_bin = analyze(x, y, c, file=file, param_dict=param_dict, return_summary_frames=True)

    #index the dataframe by the filename, sweep
    #sometimes there are no spikes, so we need to check if the dataframe has sweep Number
    if 'sweep Number' in df_raw.columns:
        df_raw = df_raw.set_index(['file_name', 'sweep Number'])
    else:
        return np.array([])

    #if we only want a subset of the features
    if feature_keys != ['']:
        df_raw = df_raw[feature_keys]
    
    if return_array:
        return df_raw.to_numpy()
    
    return df_raw


analyze_spike_times = functools.partial(analyze_template, feature_keys=subset_frames['spike_times'])
analyze_spike_times.__doc__ = analysis_temp_doc_string % 'The spike times'
analyze_spike_times.__name__ = 'analyze_spike_times'
analyze_spike_times_isi = functools.partial(analyze_template, feature_keys=subset_frames['spike_times_isi'])
analyze_spike_times_isi.__doc__ = analysis_temp_doc_string % 'The spike times and the interspike intervals'
analyze_spike_times_isi.__name__ = 'analyze_spike_times_isi'
analyze_spike_times_isi_sweepwise = functools.partial(analyze_template, feature_keys=subset_frames['spike_times_isi_sweepwise'])
analyze_spike_times_isi_sweepwise.__doc__ = analysis_temp_doc_string % 'The spike times, interspike intervals, and the sweepwise spike times'
analyze_spike_times_isi_sweepwise.__name__ = 'analyze_spike_times_isi_sweepwise'


#=== internal functions, but can be used externally ===


def process_file(file_path, param_dict, protocol_name):
    """Takes an file and runs the feature extractor on it. Filters the protocol etc.
    Essentially a wrapper for the feature extractor. As when there is an error we dont want to stop the whole program, we just want to skip the file.
    Args:
        file_path (str, os.path): _description_
        param_dict (dict): _description_
        plot_sweeps (bool): _description_
        protocol_name (str): _description_

    Returns:
        spike_dataframe, spikewise_dataframe, running_bin_data_frame : _description_
    """
    #try:
    file = cellData(file=file_path)   
    if protocol_name in file.protocol: 
        print(file_path + ' import')
        temp_spike_df, df, temp_running_bin = analyze_sweepset(file=file, sweeplist=None, param_dict=param_dict)
        return temp_spike_df, df, temp_running_bin
    else:
        print('Not correct protocol: ' + file.protocol)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    #except:
    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()


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

    #compute (or try to) some subthreshold features
    #calculate the sag
    decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, _ = exp_decay_factor(x[0], np.nanmean(y, axis=0), np.nanmean(c, axis=0), 3000)
    spike_df["Taum (Fast)"] = [decay_fast]
    spike_df["Taum (Slow)"] = [decay_slow]
    spike_df["Curve fit A"] = [curve[0]]
    spike_df["Curve fit b1"] = [curve[1]]
    spike_df["Curve fit b2"] = [curve[3]]
    spike_df["R squared 2 phase"] = [r_squared_2p]
    spike_df["R squared 1 phase"] = [r_squared_1p]
    if r_squared_2p > r_squared_1p:
        spike_df["Best Fit"] = [2]
    else:
        spike_df["Best Fit"] = [1]
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
    elif len(np.unique(stimuli_length)) == 0:
        new_current_injection_features['stimuli_length'] =  np.nan
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
    dfs = []
    averages = []
    plt.close('all')
    if (abf.sweepLabelY != 'Clamp Current (pA)' and abf.protocol != 'Gap free' and protocol_name in abf.protocol):
        np.nan_to_num(abf.data, nan=-9999, copy=False)
        if savfilter > 0:
            abf.data = signal.savgol_filter(abf.data, savfilter, polyorder=3)

        dataT = abf.sweepX
        if start_sear is not None:
            idx_start = np.argmin(np.abs(dataT - start_sear))
        else:
            idx_start = 0

        if end_sear is not None:
            idx_end = np.argmin(np.abs(dataT - end_sear))
        else:
            idx_end = -1

        if abf.sweepCount > 1:
            if subt_sweeps is None:
                sweepList = determine_subt(abf, (idx_start, idx_end))
                if np.isscalar(sweepList):
                    sweepList = np.array([sweepList])
                sweepcount = len(sweepList)
            else:
                subt_sweeps_temp = subt_sweeps - 1
                sweep_union = np.intersect1d(abf.sweepList, subt_sweeps_temp)
                sweepList = sweep_union
                sweepcount = 1
        else:
            sweepcount = 1
            sweepList = [0]

        temp_df = {}
        temp_df['filename'] = [abf.abfID]
        temp_df['foldername'] = [os.path.dirname(abf.abfFilePath)]
        temp_avg = {}
        temp_avg['filename'] = [abf.abfID]
        temp_avg['foldername'] = [os.path.dirname(abf.abfFilePath)]

        full_dataI = []
        full_dataV = []
        for sweepNumber in sweepList:
            real_sweep_length = abf.sweepLengthSec - 0.0001
            real_sweep_number = sweepNumber_to_real_sweep_number(sweepNumber) 

            abf.setSweep(sweepNumber)
            dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
            dataT, dataV, dataI = dataT[idx_start:idx_end], dataV[idx_start:idx_end], dataI[idx_start:idx_end]
            dataT = dataT - dataT[0]

            decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, p_decay = exp_decay_factor(dataT, dataV, dataI, time_after, abf_id=abf.abfID)
            resist = membrane_resistance(dataT, dataV, dataI)
            Cm2, Cm1 = mem_cap(resist, decay_slow)
            Cm3 = mem_cap_alt(resist, decay_slow, curve[3], np.amin(dataI))
            temp_df[f"_1 phase tau {real_sweep_number}"] = [p_decay]
            temp_df[f"fast 2 phase tau {real_sweep_number}"] = [decay_fast]
            temp_df[f"slow 2 phase tau {real_sweep_number}"] = [decay_slow]
            temp_df[f"Curve fit A {real_sweep_number}"] = [curve[0]]
            temp_df[f"Curve fit b1 {real_sweep_number}"] = [curve[1]]
            temp_df[f"Curve fit b2 {real_sweep_number}"] = [curve[3]]
            temp_df[f"R squared 2 phase {real_sweep_number}"] = [r_squared_2p]
            temp_df[f"R squared 1 phase {real_sweep_number}"] = [r_squared_1p]
            temp_df[f"RMP {real_sweep_number}"] = [rmp_mode(dataV, dataI)]
            temp_df[f"Membrane Resist {real_sweep_number}"] = resist / 1000000000
            temp_df[f"_2 phase Cm {real_sweep_number}"] = Cm2 * 1000000000000
            temp_df[f"_ALT_2 phase Cm {real_sweep_number}"] = Cm3 * 1000000000000
            temp_df[f"_1 phase Cm {real_sweep_number}"] = Cm1 * 1000000000000
            temp_df[f"Voltage sag {real_sweep_number}"], temp_df[f"Voltage min {real_sweep_number}"] = compute_sag(dataT, dataV, dataI, time_after, plot=bplot, clear=False)
            try:
                sag_ratio, taum_allen, voltage_allen = subthres_a(dataT, dataV, dataI, 0.0, np.amax(dataT))
                temp_df[f"Voltage sag ratio {real_sweep_number}"] = sag_ratio
                temp_df[f"Tau_m Allen {real_sweep_number}"] = taum_allen
                temp_df[f"Voltage sag Allen {real_sweep_number}"] = voltage_allen[0]
            except:
                temp_df[f"Voltage sag ratio {real_sweep_number}"] = np.nan
                temp_df[f"Tau_m Allen {real_sweep_number}"] = np.nan
                temp_df[f"Voltage sag Allen {real_sweep_number}"] = np.nan

            full_dataI.append(dataI)
            full_dataV.append(dataV)
            if dataI.shape[0] < dataV.shape[0]:
                dataI = np.hstack((dataI, np.full(dataV.shape[0] - dataI.shape[0], 0)))

        if bplot:
            plt.title(abf.abfID)
            plt.ylim(top=-40)
            plt.xlim(right=0.6)
            plt.savefig(os.path.join(os.path.dirname(abf.abfFilePath), 'cm_plots', f'sagfit{abf.abfID}sweep{real_sweep_number}.png'))

        full_dataI = np.vstack(full_dataI)
        indices_of_same = np.arange(full_dataI.shape[0])
        full_dataV = np.vstack(full_dataV)
        temp_df = pd.DataFrame.from_dict(temp_df)
        decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, p_decay = exp_decay_factor_alt(dataT, np.nanmean(full_dataV[indices_of_same, :], axis=0),
                                                                                                  np.nanmean(full_dataI[indices_of_same, :], axis=0), time_after, abf_id=abf.abfID, plot=bplot, root_fold=os.path.dirname(abf.abfFilePath))
        temp_avg[f"Voltage sag mean"], temp_avg["Voltage Min point"] = compute_sag(dataT, np.nanmean(full_dataV[indices_of_same, :], axis=0), np.nanmean(full_dataI[indices_of_same, :], axis=0), time_after, plot=bplot)
        temp_avg[f"Sweepwise Voltage sag mean"], temp_avg["Sweepwise Voltage Min point"] = np.nanmean(df_select_by_col(temp_df, ['Voltage sag 0'])), np.nanmean(df_select_by_col(temp_df, ['Voltage min']))
        if bplot:
            plt.title(abf.abfID)
            plt.savefig(os.path.join(os.path.dirname(abf.abfFilePath), 'cm_plots', f'sagfit{abf.abfID}'))

        temp_avg["Averaged 1 phase tau "] = [p_decay]
        temp_avg["Averaged 2 phase fast tau "] = [decay_fast]
        temp_avg["Averaged 2 phase slow tau "] = [decay_slow]
        temp_avg["Averaged Curve fit A"] = [curve[0]]
        temp_avg["Averaged Curve fit b1"] = [curve[1]]
        temp_avg["Averaged Curve fit b2"] = [curve[3]]
        temp_avg["Averaged R squared 2 phase"] = [r_squared_2p]
        temp_avg["Averaged R squared 1 phase"] = [r_squared_1p]
        temp_avg[f"Averaged RMP"] = [rmp_mode(np.nanmean(full_dataV[indices_of_same, :], axis=0), np.nanmean(full_dataI[indices_of_same, :], axis=0))]
        temp_avg["SweepCount Measured"] = [sweepcount]
        if r_squared_2p > r_squared_1p:
            temp_avg["Averaged Best Fit"] = [2]
        else:
            temp_avg["Averaged Best Fit"] = [1]

        resist = membrane_resistance(dataT, np.nanmean(full_dataV[indices_of_same, :], axis=0), np.nanmean(full_dataI[indices_of_same, :], axis=0))
        resist_alt = exp_rm_factor(dataT, np.nanmean(full_dataV[indices_of_same, :], axis=0), np.nanmean(full_dataI[indices_of_same, :], axis=0), time_after, decay_slow, abf_id=abf.abfID, root_fold=os.path.dirname(abf.abfFilePath))
        Cm2, Cm1 = mem_cap(resist, decay_slow, p_decay)
        Cm3 = mem_cap_alt(resist, decay_slow, curve[3], np.amin(np.nanmean(full_dataI[indices_of_same, :], axis=0)))
        rm_alt = mem_resist_alt(Cm3, decay_slow)
        temp_avg["Averaged Membrane Resist"] = resist / 1000000000
        temp_avg["Averaged Membrane Resist _ ALT"] = resist_alt[0] / 1000000000
        temp_avg["Averaged Membrane Resist _ ALT 2"] = rm_alt / 1000000000
        temp_avg["Averaged pipette Resist _ ALT"] = resist_alt[2] / 1000000000
        temp_avg["Averaged 2 phase Cm"] = Cm2 * 1000000000000
        temp_avg["Averaged 2 phase Cm Alt"] = Cm3 * 1000000000000
        temp_avg["Averaged 1 phase Cm"] = Cm1 * 1000000000000
        try:
            sag_ratio, taum_allen, voltage_allen = subthres_a(dataT, np.nanmean(full_dataV[indices_of_same, :], axis=0),
                                                              np.nanmean(full_dataI[indices_of_same, :], axis=0), 0.0, np.amax(dataT))
            temp_avg[f"Averaged Voltage sag ratio "] = sag_ratio
            temp_avg[f"Averaged Tau_m Allen "] = taum_allen
            temp_avg[f"Averaged Voltage sag min Allen "] = voltage_allen[0]
        except:
            temp_avg[f"Averaged Voltage sag ratio "] = np.nan
            temp_avg[f"Averaged Tau_m Allen "] = np.nan
            temp_avg[f"Averaged Voltage sag min Allen "] = np.nan

        try:
            mean_rms, max_rms, mean_drift, max_drift = run_qc(full_dataV[indices_of_same, :], full_dataI[indices_of_same, :])
            temp_avg["Averaged Mean RMS"] = mean_rms
            temp_avg["Max RMS"] = max_rms
            temp_avg["Averaged Mean Drift"] = mean_drift
            temp_avg["Max Drift"] = max_drift
        except:
            temp_avg["Averaged Mean RMS"] = np.nan
            temp_avg["Max RMS"] = np.nan
            temp_avg["Averaged Mean Drift"] = np.nan
            temp_avg["Max Drift"] = np.nan

        

        try:
            #here we want o use seperate subthreshold data to compute the resistance ladder
            ladder_X = []
            ladder_Y = []
            ladder_C = []
            for i in range(full_dataI.shape[0]):
                abf.setSweep(i)
                dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
                ladder_X.append(dataT)
                ladder_Y.append(dataV)
                ladder_C.append(dataI)
            rm_ladder, _, sweep_count = ladder_rm(np.vstack(ladder_X), np.vstack(ladder_Y), np.vstack(ladder_C))
            temp_avg["Resistance Ladder Slope"] = rm_ladder
            temp_avg["Rm Resistance Ladder"] = 1 / rm_ladder
            temp_avg["Resistance Ladder SweepCount Measured"] = sweep_count
        except:
            temp_avg["Resistance Ladder Slope"] = np.nan
            temp_avg["Rm Resistance Ladder"] = np.nan
            temp_avg["Resistance Ladder SweepCount Measured"] = np.nan

        temp_avg = pd.DataFrame.from_dict(temp_avg).T
        #now we can append the dataframes
        temp_avg = _merge_current_injection_features(sweepX=np.tile(dataT, (full_dataI.shape[0], 1)), sweepY=full_dataI, sweepC=full_dataI, spike_df=temp_avg)

        
        dfs = temp_df
        averages = temp_avg

    return dfs, averages


### IPFX FIXES
# here we have a few functions that are used to fix the IPFX package, since there is a few errors
# in the original code. The functions are copied from the IPFX package and modified to work with
# the current version of the package. The functions are:
# find_downstroke_indexes

def find_downstroke_indexes(v, t, peak_indexes, trough_indexes, clipped=None, filter=10., dvdt=None):
    """Find indexes of minimum voltage (troughs) between spikes.

    Parameters
    ----------
    v : numpy array of voltage time series in mV
    t : numpy array of times in seconds
    peak_indexes : numpy array of spike peak indexes
    trough_indexes : numpy array of threshold indexes
    clipped: boolean array - False if spike not clipped by edge of window
    filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
    dvdt : pre-calculated time-derivative of voltage (optional)

    Returns
    -------
    downstroke_indexes : numpy array of downstroke indexes
    """

    if not trough_indexes.size:
        return np.array([])

    if dvdt is None:
        dvdt = tsu.calculate_dvdt(v, t, filter)

    if clipped is None:
        clipped = np.zeros_like(peak_indexes, dtype=bool)

    if len(peak_indexes) < len(trough_indexes):
        raise er.FeatureError("Cannot have more troughs than peaks")
    # Taking this out...with clipped info, should always have the same number of points
    #     peak_indexes = peak_indexes[:len(trough_indexes)]

    valid_peak_indexes = peak_indexes[~clipped].astype(int)
    valid_trough_indexes = trough_indexes[~clipped].astype(int)

    downstroke_indexes = np.zeros_like(peak_indexes) * np.nan

    #handle argmin of empty array
    zipped_idxs = []
    for i,j in zip(valid_peak_indexes, valid_trough_indexes):
        #if j is less than i just add one
        if j <= i:
            j = i+1
        zipped_idxs.append((i, j))

    downstroke_index_values = [np.argmin(dvdt[peak:trough]) + peak for peak, trough
                         in zipped_idxs]
    downstroke_indexes[~clipped] = downstroke_index_values

    return downstroke_indexes

#override
ipfx.spike_detector.find_downstroke_indexes = find_downstroke_indexes


from ipfx import spike_detector,time_series_utils
def determine_rejected_spikes(spfx, spike_df, v, t, param_dict):
    """Determine which spikes were rejected by the spike detection algorithm.
    Parameters
    ----------
    spfx : SweepFeatures object
    spike_df : pandas.DataFrame
        DataFrame containing spike features
    Returns
    -------
    rejected_spikes : list of bool
        True if spike was rejected, False if spike was accepted
    """
    dvdt = time_series_utils.calculate_dvdt(v, t, 0)

    rejected_spikes = {}
    intial_spikes = spike_detector.detect_putative_spikes(v, t, param_dict['start'], param_dict['end'],
                                                    dv_cutoff=param_dict['dv_cutoff'],
                                                    dvdt=dvdt)
    peaks = spike_detector.find_peak_indexes(v, t, intial_spikes, param_dict['end'])
    if len(peaks) == 0:
        return rejected_spikes
    diff_mask = [np.any(dvdt[peak_ind:spike_ind] < 0)
                 for peak_ind, spike_ind
                 in zip(peaks[:-1], intial_spikes[1:])]
    peak_indexes = peaks[np.array(diff_mask + [True])]
    spike_indexes = intial_spikes[np.array([True] + diff_mask)]

    peak_level_mask = v[peak_indexes] >= param_dict['min_peak']
        

    height_mask = (v[peak_indexes] - v[spike_indexes]) >= param_dict['min_height']
    for i, spike in enumerate(peak_indexes):
        if np.any([~peak_level_mask[i], ~height_mask[i]]):
            rejected_spikes[spike] = {'peak_level': ~peak_level_mask[i], 'height': height_mask[i]}
    
    peak_level_mask = v[peak_indexes] >= param_dict['min_peak']
    spike_indexes = spike_indexes[peak_level_mask]
    peak_indexes = peak_indexes[peak_level_mask]

    height_mask = (v[peak_indexes] - v[spike_indexes]) >= param_dict['min_height']
    spike_indexes = spike_indexes[height_mask]
    peak_indexes = peak_indexes[height_mask]
    
    if len(spike_indexes) == 0:
        return rejected_spikes
    upstroke_indexes = spike_detector.find_upstroke_indexes(v, t, spike_indexes, peak_indexes, filter=0, dvdt=dvdt)
    thresholds = spike_detector.refine_threshold_indexes(v, t, upstroke_indexes, param_dict['thresh_frac'],
                                                dvdt=dvdt)


    # overlaps = np.flatnonzero(spike_indexes[1:] <= peak_indexes[:-1] + 1)
    # if overlaps.size:
    #     spike_mask = np.ones_like(spike_indexes, dtype=bool)
    #     spike_mask[overlaps + 1] = False
    #     spike_indexes = spike_indexes[spike_mask]

    #     peak_mask = np.ones_like(peak_indexes, dtype=bool)
    #     peak_mask[overlaps] = False
    #     peak_indexes = peak_indexes[peak_mask]

    #     upstroke_mask = np.ones_like(upstroke_indexes, dtype=bool)
    #     upstroke_mask[overlaps] = False
    #     upstroke_indexes = upstroke_indexes[upstroke_mask]

    # Validate that peaks don't occur too long after the threshold
    # If they do, try to re-find threshold from the peak
    too_long_spikes = []
    for i, (spk, peak) in enumerate(zip(spike_indexes, peak_indexes)):
        if t[peak] - t[spk] >= param_dict['max_interval']:
            too_long_spikes.append(i)
    if too_long_spikes:
        i
        avg_upstroke = dvdt[upstroke_indexes].mean()
        target = avg_upstroke * param_dict['thresh_frac']
        drop_spikes = []
        for i in too_long_spikes:
            # First guessing that threshold is wrong and peak is right
            peak = peak_indexes[i]
            t_0 = time_series_utils.find_time_index(t, t[peak] - param_dict['max_interval'])
            below_target = np.flatnonzero(dvdt[upstroke_indexes[i]:t_0:-1] <= target)
            if not below_target.size:
                # Now try to see if threshold was right but peak was wrong

                # Find the peak in a window twice the size of our allowed window
                spike = spike_indexes[i]
                t_0 = time_series_utils.find_time_index(t, t[spike] + 2 * param_dict['max_interval'])
                new_peak = np.argmax(v[spike:t_0]) + spike

                # If that peak is okay (not outside the allowed window, not past the next spike)
                # then keep it
                if t[new_peak] - t[spike] < param_dict['max_interval'] and \
                   (i == len(spike_indexes) - 1 or t[new_peak] < t[spike_indexes[i + 1]]):
                    peak_indexes[i] = new_peak
                else:
                    # Otherwise, log and get rid of the spike
                    drop_spikes.append(i)
            else:
                spike_indexes[i] = upstroke_indexes[i] - below_target[0]
        for i in drop_spikes:
            rejected_spikes[spike_indexes[i]] = {'peak_level': False, 'height': False, 'threshold to peak': True, }
    else:
        return rejected_spikes
    return rejected_spikes