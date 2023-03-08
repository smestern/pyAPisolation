"""This feature extractor module is a wrapper for the feature extractor class found in ipfx,
    it takes an abf file and runs the feature extractor on it. Filters the ABF by protocol etc.
    Essentially a wrapper for the feature extractor. As when there is an error we dont want to stop the whole program, we just want to skip the abf.
    
    Since code here is a bit messy, I have broken it up into a few functions.

"""

# imports
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf
import copy
import multiprocessing as mp
from ipfx import feature_extractor
from ipfx import subthresh_features as subt
import scipy.signal as signal
import logging
logging.basicConfig(level=logging.INFO)

from .loadABF import loadABF
from .patch_utils import plotabf, load_protocols, find_non_zero_range, filter_abf, build_running_bin
from .patch_subthres import *
from .QC import run_qc

logging.debug("Feature extractor loaded")

#GLOBALS
parallel = True
default_dict = {'start': 0, 'end': 0, 'filter': 0, 'stim_find': True}
#Labels found in the spike train dataframe produced by ipfx
ipfx_train_feature_labels =['adapt', 'latency', 'isi_cv', 'mean_isi', 'median_isi', 'first_isi',
       'avg_rate']

#labels used to build the running average dataframe
running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width', 'isi']

#labels to include in the excel subsheets
subsheets_spike = {'spike count':['spike count'], 'rheobase features':['rheobase'], 
                    'mean':['mean'], 'isi':['isi'], 'latency': ['latency_'], 'current':['current'],'QC':['QC'], 
                    'spike features':['spike_'], 'subthres features':['baseline voltage', 'Sag', 'Taum'], 'full sheet': ['']}


def folder_feature_extract(files, param_dict, plot_sweeps=-1, protocol_name='IC1', para=1):
    """runs the feature extractor on a folder of abfs.

    Args:
        files (list): _description_
        param_dict (dict): _description_
        plot_sweeps (int, bool, optional): _description_. Defaults to -1.
        protocol_name (str, optional): _description_. Defaults to 'IC1'.
        para (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    dfs = pd.DataFrame()
    df_spike_count = pd.DataFrame()
    df_running_avg_count = pd.DataFrame()
    filelist = glob.glob(files + "\\**\\*.abf", recursive=True)
    spike_count = []
    df_full = []
    df_running_avg = []
    if parallel:
        pool = mp.Pool()
        results = [pool.apply(preprocess_abf, args=(file, param_dict, plot_sweeps, protocol_name)) for file in filelist]
        pool.close()
        ##split out the results
        for result in results:
            if result[0].empty:
                print('Empty result')
    else:
        for f in filelist:
            temp_df_spike_count, temp_full_df, temp_running_bin = preprocess_abf(f, copy.deepcopy(param_dict), plot_sweeps, protocol_name)
            spike_count.append(temp_df_spike_count)
            df_full.append(temp_full_df)
            df_running_avg.append(temp_running_bin)
    df_spike_count = pd.concat(spike_count, sort=True)
    dfs = pd.concat(df_full, sort=True)
    df_running_avg_count = pd.concat(df_running_avg, sort=False)
    return dfs, df_spike_count, df_running_avg_count

def preprocess_abf(file_path, param_dict, plot_sweeps, protocol_name):
    """Takes an abf file and runs the feature extractor on it. Filters the ABF by protocol etc.
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
        abf = pyabf.ABF(file_path)           
        if abf.sweepLabelY != 'Clamp Current (pA)' and protocol_name in abf.protocol:
            print(file_path + ' import')

            temp_spike_df, df, temp_running_bin = analyze_abf(abf, sweeplist=None, plot=plot_sweeps, param_dict=param_dict)
            return temp_spike_df, df, temp_running_bin
        else:
            print('Not correct protocol: ' + abf.protocol)
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    except:
       return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

def analyze_spike_sweep(dataT, dataV, dataI, param_dict):
    """analyzes a single sweep of data for spikes

    Args:
        dataT (_type_): 1D array of time values (in seconds)
        dataV (_type_): 1D array of voltage values (in mV)
        dataI (_type_): 1D array of current values (in pA); if None, then current is assumed to be zero
        param_dict (_type_): Parameter dict

    Returns:
        _type_: _description_
    """    """"""
    spikext = feature_extractor.SpikeFeatureExtractor(**param_dict)
    spiketxt = feature_extractor.SpikeTrainFeatureExtractor(start=param_dict['start'], end=param_dict['end'])  
    spike_in_sweep = spikext.process(dataT, dataV, dataI)
    spike_train = spiketxt.process(dataT, dataV, dataI, spike_in_sweep)
    return spike_in_sweep, spike_train


def analyze_abf(abf, sweeplist=None, plot=-1, param_dict=None):
    """A semi-automated analysis of an abf file, includes spike detection In most cases, requires no user input.
    However this is dependent on the protocol and the user may need to adjust the parameters.

    Args:
        abf (_type_): The abf file object to analyze
        sweeplist (list, optional): The sweeps to analyze. Defaults to None, which means all sweeps.
        plot (int, optional): _description_. Defaults to -1.
        param_dict (_type_, optional): _description_. Defaults to None.

    Returns:
        filewise_dataframe: A dataframe of the features, with each file as a separate row.
        spike_dataframe: A dataframe of the spike features, with each spike as a separate row.
        running_bin_data_frame: A dataframe of the running bin features, with each bin as a separate row.
    """    
    #deals with some nan issues in the abf
    #ipfx throws a warning if the abf has a nan value in the voltage trace
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
    stim_find = param_dict.pop('stim_find')
    #for now if user wants to filter by stim time we will just use the first sweep
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
    #load the ABF FILE once
    x, y ,c = loadABF(abf.abfFilePath)
    #iterate through the sweeps
    for sweepNumber in sweepcount: 
        #set the current sweep
        abf.setSweep(sweepNumber)
        dataT, dataV, dataI = x[sweepNumber], y[sweepNumber], c[sweepNumber]
        #if the user asks for a filter, apply it
        if bessel_filter is not None:
            if bessel_filter != -1:
                dataV = filter_abf(dataV, abf, bessel_filter)
        if dataI.shape[0] < dataV.shape[0]:
                    dataI = np.hstack((dataI, np.full(dataV.shape[0] - dataI.shape[0], 0)))
        real_sweep_length = abf.sweepLengthSec - 0.0001 #hack to prevent issues with rounding
        #set the 'real_sweep_number' for easy sorting with pandas
        if sweepNumber < 9:
            real_sweep_number = '00' + str(sweepNumber + 1)
        elif sweepNumber > 8 and sweepNumber < 99:
            real_sweep_number = '0' + str(sweepNumber + 1)
        #ensure the user doesnt ask for a time period that is too long
        if param_dict['start'] == 0 and param_dict['end'] == 0: 
            param_dict['end']= real_sweep_length
        elif param_dict['end'] > real_sweep_length:
            param_dict['end'] = real_sweep_length
        #analyze the sweep
        #analyze the subthreshold features
        subthres_df = analyze_subthreshold(dataT, dataV, dataI, real_sweep_number, param_dict)
        spike_in_sweep, spike_train = analyze_spike_sweep(dataT, dataV, dataI, param_dict) ### Returns the default Dataframe Returned by ipfx
        #now we need to transform the spike_in_sweep dataframe into a dataframe that is easier to work with
        #here we pass back and fourth the spike_in_sweep and spike_train dataframes
        temp_spike_df, df, temp_running_bin = _build_sweepwise_dataframe(real_sweep_number, spike_in_sweep, spike_train, temp_spike_df, df, temp_running_bin, param_dict)
    #once we have compiled the dataframes we need to compute some final features
    temp_spike_df, df, temp_running_bin = _build_full_df(abf, temp_spike_df, df, temp_running_bin, sweepcount)
    #try qc or just return the dataframe
    try:
        _qc_data = run_qc(y, c)
        temp_spike_df['QC Mean RMS'] = _qc_data[0]
        temp_spike_df['QC Mean Sweep Drift'] = _qc_data[2]
    except:
        temp_spike_df['QC Mean RMS'] = np.nan
        temp_spike_df['QC Mean Sweep Drift'] = np.nan
    #if the user wants to plot, do it:
    try:
        spiketimes = np.transpose(np.vstack((np.ravel(df['peak_index'].to_numpy()), np.ravel(df['sweep Number'].to_numpy()))))
        plotabf(abf, spiketimes, param_dict['start'], param_dict['end'], plot)
    except:
        pass
    return temp_spike_df, df, temp_running_bin

def preprocess_abf_subthreshold(file_path, protocol_name='', param_dict={}):
    try:
        abf = pyabf.ABF(file_path)           
        if abf.sweepLabelY != 'Clamp Current (pA)' and protocol_name in abf.protocol:
            print(file_path + ' import')

            df, avg = analyze_cm(abf, sweeplist=None,  **param_dict)
            return df, avg
        else:
            print('Not correct protocol: ' + abf.protocol)
            return pd.DataFrame(), pd.DataFrame()
    except:
       return pd.DataFrame(), pd.DataFrame()

def analyze_subthreshold(dataT, dataV, dataI, sweep_number, param_dict):
    """
    Analyzes the subthreshold features of an ABF file.
    Args:
        dataT: The time trace of the ABF file.
        dataV: The voltage trace of the ABF file.
        dataI: The current trace of the ABF file.
    Returns:
        A dataframe of the subthreshold features.
    """
    dict_data = {}
    #find the subthreshold features
    #compute the subthreshold features #TODO MOVE THIS TO A SEPARATE FUNCTION
    try:
        dict_data['baseline voltage' + sweep_number] = subt.baseline_voltage(dataT,dataV, start=0.1, filter_frequency=param_dict['filter'])
    except:
        print('Fail to find baseline voltage')
    sag, taum, voltage = subthres_a(dataT, dataV, dataI, param_dict['start'], param_dict['end'])
    dict_data["Sag Ratio " + sweep_number + ""] = sag
    dict_data["Taum " + sweep_number + ""] = taum 
    #TODO move this to a separate function This function should be able to only use dataframes
    curve = exp_growth_factor(dataT, dataV, dataI, int(10))
    dict_data["exp growth tau1" + sweep_number] = curve[2]
    dict_data["exp growth tau2" + sweep_number] = curve[-1]   
    return pd.DataFrame(dict_data)

def save_data_frames(dfs, df_spike_count, df_running_avg_count, root_fold='', tag=''):
    """_summary_

    Args:
        dfs (_type_): _description_
        df_spike_count (_type_): _description_
        df_running_avg_count (_type_): _description_
        root_fold (str, optional): _description_. Defaults to ''.
        tag (str, optional): _description_. Defaults to ''.
    """
    #try and save the dataframes to csv files. try and catch for each dataframe separately
    try:
        dfs.to_csv(root_fold + '/allfeatures_' + tag + '.csv')
    except:
        print('Could not save all features dataframe')
    try:
        with pd.ExcelWriter(root_fold + '/spike_count_' + tag + '.xlsx') as runf:
            cols = df_spike_count.columns.values
            df_ind = df_select_by_col(df_spike_count, ['foldername', 'filename'])
            df_ind = df_ind.loc[:,['foldername', 'filename']]
            index = pd.MultiIndex.from_frame(df_ind)
            for key, p in subsheets_spike.items():
                temp_ind = df_select_by_col(df_spike_count, p)
                temp_df = temp_ind.set_index(index)
                temp_df.to_excel(runf, sheet_name=key)
            #print(df_ind)\
    except:
        print('Could not save spike count dataframe')
    
    try:
        with pd.ExcelWriter(root_fold + '/running_avg_' + tag + '.xlsx') as runf:
            cols = df_running_avg_count.columns.values
            df_ind = df_running_avg_count.loc[:,['foldername', 'filename', 'Sweep Number']]
            index = pd.MultiIndex.from_frame(df_ind)
            for p in running_lab:
                temp_ind = [p in col for col in cols]
                temp_df = df_running_avg_count.set_index(index).loc[:,temp_ind]
                temp_df.to_excel(runf, sheet_name=p)
    except:
        print('Could not save running average dataframe')

    #try:  
        #ids = dfs['file_name'].unique()
        #tempframe = dfs.groupby('file_name').mean().reset_index()
        #tempframe.to_csv(root_fold + '/allAVG_' + tag + '.csv')
        #tempframe = dfs.drop_duplicates(subset='file_name')
        #tempframe.to_csv(root_fold + '/allRheo_' + tag + '.csv')
        
        #df_spike_count.to_csv(root_fold + '/spike_count_' + tag + '.csv')
    #except: 
        #print('error saving')

def _build_sweepwise_dataframe(real_sweep_number, spike_in_sweep, spike_train, temp_spike_df, df, temp_running_bin, param_dict):
    """ This function is used to build the dataframe for a single sweep. 
    We calculate some sweepwise features from the ipfx output and affix them to the dataframe.
    
    Args:
        real_sweep_number (int): the real sweep number of the abf file (the python index + 1)
        spike_in_sweep (list): list of the spike indices in the sweep
        spike_train (list): list of the spike times in the sweep
        temp_spike_df (dataframe): dataframe of the spike features
        df (dataframe): dataframe of the features
        temp_running_bin (dataframe): dataframe of the running average features
        param_dict (dict): dictionary of the parameters
    Returns:
        df (dataframe): dataframe of the features
        temp_running_bin (dataframe): dataframe of the running average features """
    #first count the number of spikes in the sweep
    spike_count = spike_in_sweep.shape[0]
    temp_spike_df["Sweep " + real_sweep_number + " spike count"] = [spike_count]
    #Now compute the current injected this sweep (really only useful for square pulses)
    current_str = np.array2string(np.unique(abf.sweepC))
    current_str = current_str.replace('[', '')
    current_str = current_str.replace(' 0.', '')
    current_str = current_str.replace(']', '')
    temp_spike_df["Current_Sweep " + real_sweep_number + " current injection"] = [current_str]

    #Now prep the running average dataframe, here we are generating the column names and the index
    sweep_running_bin = pd.DataFrame()
    time_bins = np.arange(param_dict['start']*1000, param_dict['end']*1000+20, 20)
    _run_labels = []
    for p in running_lab:
            temp_lab = []
            for x in time_bins :
                temp_lab = np.hstack((temp_lab, f'{p} {x} bin AVG'))
            _run_labels.append(temp_lab)
    _run_labels = np.hstack(_run_labels).tolist()
    #create a nan row in case we have no spikes
    nan_row_run = np.ravel(np.full((len(running_lab), time_bins.shape[0]), np.nan)).reshape(1,-1)


    if spike_count > 0:
        #if we have more than one spike this sweep we can compute various features
        #in this case we compute the spike features
        trough_averge,_  = build_running_bin(spike_in_sweep['fast_trough_v'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])
        peak_average = build_running_bin(spike_in_sweep['peak_v'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
        peak_max_rise = build_running_bin(spike_in_sweep['upstroke'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
        peak_max_down = build_running_bin(spike_in_sweep['downstroke'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
        peak_width = build_running_bin(spike_in_sweep['width'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
        isi_bin = build_running_bin(np.diff(spike_in_sweep['peak_t'].to_numpy()), spike_in_sweep['peak_t'].to_numpy()[:-1], start=param_dict['start'], end=param_dict['end'])[0]        
        sweep_running_bin = pd.DataFrame(data=np.hstack((trough_averge, peak_average, peak_max_rise, peak_max_down, peak_width, isi_bin)).reshape(1,-1), columns=_run_labels, index=[real_sweep_number])

    
        #now we compute the spike features, here we are taking spike features from ipfx and manipulating them
        #to make them more useful for the analysis
        #generally these features are the same as the features in the spike dataframe, however only pulling the first spike each sweep
        temp_spike_df["first_isi " + real_sweep_number + " isi"] = [spike_train['first_isi']]
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


        
                
        if spike_count > 2:
            #if we have more than two spikes we can compute more features
            f_isi = spike_in_sweep['peak_t'].to_numpy()[-1] #the last ISI
            l_isi = spike_in_sweep['peak_t'].to_numpy()[-2] #the second last ISI
            temp_spike_df["last_isi" + real_sweep_number + " isi"] = [abs( f_isi- l_isi )] #the difference between the last two ISI
            spike_in_sweep['isi_'] = np.hstack((np.diff(spike_in_sweep['peak_t'].to_numpy()), np.nan)) #the difference between the last two ISI
            temp_spike_df["min_isi" + real_sweep_number + " isi"] = np.nanmin(np.hstack((np.diff(spike_in_sweep['peak_t'].to_numpy()), np.nan))) #the min isi

            #temp_spike_df["spike_" + real_sweep_number + " 2"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[1] - spike_in_sweep['threshold_v'].to_numpy()[1])
            #temp_spike_df["spike_" + real_sweep_number + " 3"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[-1] - spike_in_sweep['threshold_v'].to_numpy()[-1])
            #temp_spike_df["spike_" + real_sweep_number + "AHP 2"] = spike_in_sweep['fast_trough_v'].to_numpy()[1]
            #temp_spike_df["spike_" + real_sweep_number + "AHP 3"] = spike_in_sweep['fast_trough_v'].to_numpy()[-1]
            #temp_spike_df["spike_" + real_sweep_number + "AHP height 2"] = abs(spike_in_sweep['peak_v'].to_numpy()[1] - spike_in_sweep['fast_trough_v'].to_numpy()[1])
            #temp_spike_df["spike_" + real_sweep_number + "AHP height 3"] = abs(spike_in_sweep['peak_v'].to_numpy()[-1] - spike_in_sweep['fast_trough_v'].to_numpy()[-1])
            #temp_spike_df["spike_width" + real_sweep_number + "2"] = spike_in_sweep['width'].to_numpy()[1]
            #temp_spike_df["spike_width" + real_sweep_number + "3"] = spike_in_sweep['width'].to_numpy()[-1]
            for label in ipfx_train_feature_labels:
                #for the spike train features from ipfx, we just need to affix them with thier label
                try:
                    temp_spike_df[label + real_sweep_number] = spike_train[label]
                except:
                    #if the feature is not in the spike train, we just leave it as nan
                    temp_spike_df[label + real_sweep_number] = np.nan
            
        else:
            #if we have only one spike we can't compute more features, so we affix them as nan
            temp_spike_df["last_isi" + real_sweep_number + " isi"] = [np.nan]
            spike_in_sweep['isi_'] = np.hstack((np.full(abs(spike_count), np.nan)))
            temp_spike_df["min_isi" + real_sweep_number + " isi"] = [spike_train['first_isi']] #except this one we can just use the first ISI
            for label in ipfx_train_feature_labels:
                temp_spike_df[label + real_sweep_number] = [np.nan]
        #debugging statement
        print("Processed Sweep " + str(real_sweep_number) + " with " + str(spike_count) + " aps")
        #affix the sweep spikes to the full featured dataframe
        df = df.append(spike_in_sweep, ignore_index=True, sort=True)
    else:
        #if we don't have any spikes we can't compute more features, so we affix them as nan
        #TODO this is a bit of a hack, we should put this in a loop or something
        temp_spike_df["latency_" + real_sweep_number + " latency"] = [np.nan]
        temp_spike_df["first_isi " + real_sweep_number + " isi"] = [np.nan]
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
        temp_spike_df["exp growth tau1" + real_sweep_number] = [np.nan]
        temp_spike_df["exp growth tau2" + real_sweep_number] = [np.nan]
        for label in ipfx_train_feature_labels:
            temp_spike_df[label + real_sweep_number] = [np.nan]
    #affix the sweep spikes to the full featured dataframe
    #add the abf ID and the sweep number to the dataframe
    sweep_running_bin['Sweep Number'] = [real_sweep_number]
    sweep_running_bin['filename'] = [abf.abfID]
    sweep_running_bin['foldername'] = [os.path.dirname(abf.abfFilePath)]

    
    temp_running_bin = temp_running_bin.append(sweep_running_bin)
    return temp_spike_df, df, temp_running_bin


def _build_full_df(abf, temp_spike_df, df, temp_running_bin, sweepList):
    """ Once all the sweeps are processed, we can build the full dataframe. This
    essentially computes the means and rheobase features.

    Args:
        abf (_type_): _description_
        temp_spike_df (_type_): _description_
        df (_type_): _description_
        temp_running_bin (_type_): _description_
        sweepList (_type_): _description_

    Returns:
        _type_: _description_
    """
    temp_spike_df['protocol'] = [abf.protocol]
    if df.empty:
        df = df.assign(file_name=np.full(1,abf.abfID))
        df = df.assign(__fold_name=np.full(1,os.path.dirname(abf.abfFilePath)))
        print('no spikes found')
    else:
        df = df.assign(file_name=np.full(len(df.index),abf.abfID))
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
        temp_spike_df["rheobase_downstroke"] = [df['downstroke'].to_numpy()[0]]
        temp_spike_df["rheobase_fast_trough"] = [df['fast_trough_v'].to_numpy()[0]]
        for key in ipfx_train_feature_labels:
            temp_spike_df[f"mean_{key}"] = [np.nanmean(df[key].to_numpy())]
        temp_spike_df["mean_current"] = [np.nanmean(df['peak_i'].to_numpy())]
        temp_spike_df["mean_latency"] = [np.nanmean(df['latency'].to_numpy())]
        temp_spike_df["mean_thres"] = [np.nanmean(df['threshold_v'].to_numpy())]
        temp_spike_df["mean_width"] = [np.nanmean(df['width'].to_numpy())]
        temp_spike_df["mean_heightPT"] = [np.nanmean(abs(df['peak_v'].to_numpy() - df['fast_trough_v'].to_numpy()))]
        temp_spike_df["mean_heightTP"] = [np.nanmean(abs(df['threshold_v'].to_numpy() - df['peak_v'].to_numpy()))]
        temp_spike_df["mean_upstroke"] = [np.nanmean(df['upstroke'].to_numpy())]
        temp_spike_df["mean_downstroke"] = [np.nanmean(df['downstroke'].to_numpy())]
        temp_spike_df["mean_fast_trough"] = [np.nanmean(df['fast_trough_v'].to_numpy())]
        spiketimes = np.transpose(np.vstack((np.ravel(df['peak_index'].to_numpy()), np.ravel(df['sweep Number'].to_numpy()))))
        

    full_dataI = []
    full_dataV = []
    full_subt = []
    for x in sweepList:
        abf.setSweep(x)
        full_dataI.append(abf.sweepC)
        full_dataV.append(abf.sweepY)
    full_dataI = np.vstack(full_dataI) 
    full_dataV = np.vstack(full_dataV) 
    decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, _ = exp_decay_factor(abf.sweepX, np.nanmean(full_dataV,axis=0), np.nanmean(full_dataI,axis=0), 3000, abf_id=abf.abfID)
    temp_spike_df["Taum (Fast)"] = [decay_fast]
    temp_spike_df["Taum (Slow)"] = [decay_slow]
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
                


def analyze_cm(abf, protocol_name='', savfilter=0, start_sear=None, end_sear=None, subt_sweeps=None, time_after=50, bplot=False):
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
    temp_df['1Afilename'] = [abf.abfID]
    temp_df['1Afoldername'] = [os.path.dirname(file_path)]
    temp_avg = pd.DataFrame()
    temp_avg['1Afilename'] = [abf.abfID]
    temp_avg['1Afoldername'] = [os.path.dirname(file_path)]
    
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
    




class patchFeatExtractor(object):
    """TODO """
    def __init__(self, file, start=None, end=None, filter=10.,
                 dv_cutoff=20., max_interval=0.005, min_height=2., min_peak=-30.,
                 thresh_frac=0.05, reject_at_stim_start_interval=0):
        """Initialize SweepFeatures object.-
        Parameters
        ----------
        start : start of time window for feature analysis (optional)
        end : end of time window for feature analysis (optional)
        filter : cutoff frequency for 4-pole low-pass Bessel filter in kHz (optional, default 10)
        dv_cutoff : minimum dV/dt to qualify as a spike in V/s (optional, default 20)
        max_interval : maximum acceptable time between start of spike and time of peak in sec (optional, default 0.005)
        min_height : minimum acceptable height from threshold to peak in mV (optional, default 2)
        min_peak : minimum acceptable absolute peak level in mV (optional, default -30)
        thresh_frac : fraction of average upstroke for threshold calculation (optional, default 0.05)
        reject_at_stim_start_interval : duration of window after start to reject potential spikes (optional, default 0)
        """
        if isinstance(abf, pyabf.ABF):
            self.abf = abf
        elif isinstance(abf, str) or isinstance(abf, os.path.abspath):
            self.abf = pyabf.ABF
        self.start = start
        self.end = end
        self.filter = filter
        self.dv_cutoff = dv_cutoff
        self.max_interval = max_interval
        self.min_height = min_height
        self.min_peak = min_peak
        self.thresh_frac = thresh_frac
        self.reject_at_stim_start_interval = reject_at_stim_start_interval
        self.spikefeatureextractor = feature_extractor.SpikeFeatureExtractor(start=start, end=end, filter=filter, dv_cutoff=dv_cutoff, max_interval=max_interval, min_height=min_height, min_peak=min_peak, thresh_frac=thresh_frac, reject_at_stim_start_interval=reject_at_stim_start_interval)
        self.spiketrainextractor = feature_extractor.SpikeTrainFeatureExtractor(start=start, end=end)

if __name__ == '__main__':
    mp.freeze_support()