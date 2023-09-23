#############
# this script is built to do some additional processing on the output of ipfx
# primairly we will be computing some features needed / desired for the inoue lab

# here we will be wrangling some of the data frames and saving them to csv files
# we will also be generating some plots and saving them to the same folder
#############
import numpy as np
import os
import pandas as pd

from .patch_utils import *
from .patch_subthres import *

ipfx_train_feature_labels =['adapt', 'latency', 'isi_cv', 'mean_isi', 'median_isi', 'first_isi',
       'avg_rate']
running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width', 'isi']
subsheets_spike = {'spike count':['spike count'], 'rheobase features':['rheobase'], 
                    'mean':['mean'], 'isi':['isi'], 'latency': ['latency_'], 'current':['current'],'QC':['QC'], 
                    'spike features':['spike_'], 'subthres features':['baseline voltage', 'Sag', 'Taum'], 'full sheet': ['']}
def save_data_frames(dfs, df_spike_count, df_running_avg_count, root_fold='', tag=''):
    #try:  
    #ids = dfs['file_name'].unique()
    #tempframe = dfs.groupby('file_name').mean().reset_index()
    #tempframe.to_csv(root_fold + '/allAVG_' + tag + '.csv')
    #tempframe = dfs.drop_duplicates(subset='file_name')
    #tempframe.to_csv(root_fold + '/allRheo_' + tag + '.csv')
    
    #df_spike_count.to_csv(root_fold + '/spike_count_' + tag + '.csv')
    dfs.to_csv(root_fold + '/allfeatures_' + tag + '.csv')
    with pd.ExcelWriter(root_fold + '/spike_count_' + tag + '.xlsx') as runf:
        cols = df_spike_count.columns.values
        df_ind = df_select_by_col(df_spike_count, ['foldername', 'filename'])
        df_ind = df_ind.loc[:,['foldername', 'filename']]
        index = pd.MultiIndex.from_frame(df_ind)
        for key, p in subsheets_spike.items():
            temp_ind = df_select_by_col(df_spike_count, p)
            temp_df = temp_ind.set_index(index)
            temp_df.to_excel(runf, sheet_name=key)
        #print(df_ind)
    
    with pd.ExcelWriter(root_fold + '/running_avg_' + tag + '.xlsx') as runf:
        cols = df_running_avg_count.columns.values
        df_ind = df_running_avg_count.loc[:,['foldername', 'filename', 'Sweep Number']]
        index = pd.MultiIndex.from_frame(df_ind)
        for p in running_lab:
            temp_ind = [p in col for col in cols]
            temp_df = df_running_avg_count.set_index(index).loc[:,temp_ind]
            temp_df.to_excel(runf, sheet_name=p)
    #except: 
        #print('error saving')


# TODO <--- this is a mess, clean it up
# Ensure functions do no require the abf object, or dataframes
# functions should not depend on further analysis, only concat of dataframes




def _build_sweepwise_dataframe(abf, real_sweep_number, spike_in_sweep, spike_train, temp_spike_df, df, temp_running_bin, param_dict):
    """Build a "sweepwise" dataframe for a single sweep. Essentialy compacts the features of a single sweep into a singluar columns of a dataframe
    Args:
        abf (_type_): _description_
        real_sweep_number (_type_): _description_
        spike_in_sweep (_type_): ipfx output spike dataframe for a single sweep
        spike_train (_type_): ipfx output spike train dataframe for a single sweep
        temp_spike_df (_type_): 
        df (_type_): _description_
        temp_running_bin (_type_): _description_
        param_dict (_type_): _description_

    Returns:
        _type_: _description_
    """
    #first declare some dicts to hold the data, they will be converted to dataframes later
    dict_spike_df = {}



    spike_count = spike_in_sweep.shape[0]
    dict_spike_df["Sweep " + real_sweep_number + " spike count"] = [spike_count]
    current_str = np.array2string(np.unique(abf.sweepC))
    current_str = current_str.replace('[', '')
    current_str = current_str.replace(' 0.', '')
    current_str = current_str.replace(']', '')
    sweep_running_bin = pd.DataFrame()
    dict_spike_df["Current_Sweep " + real_sweep_number + " current injection"] = [current_str]
    time_bins = np.arange(param_dict['start']*1000, param_dict['end']*1000+20, 20)
    _run_labels = []
    for p in running_lab:
            temp_lab = []
            for x in time_bins :
                temp_lab = np.hstack((temp_lab, f'{p} {x} bin AVG'))
            _run_labels.append(temp_lab)
    _run_labels = np.hstack(_run_labels).tolist()
    nan_row_run = np.ravel(np.full((len(running_lab), time_bins.shape[0]), np.nan)).reshape(1,-1)
    try:
        dict_spike_df['baseline voltage' + real_sweep_number] = subt.baseline_voltage(abf.sweepX, abf.sweepY, start=0.1, filter_frequency=param_dict['filter'])
    except:
        print('Fail to find baseline voltage')
            

    if spike_count > 0:
        dict_spike_df["first_isi " + real_sweep_number + " isi"] = [spike_train['first_isi']]
        trough_averge,_  = build_running_bin(spike_in_sweep['fast_trough_v'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])
        peak_average = build_running_bin(spike_in_sweep['peak_v'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
        peak_max_rise = build_running_bin(spike_in_sweep['upstroke'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
        peak_max_down = build_running_bin(spike_in_sweep['downstroke'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
        peak_width = build_running_bin(spike_in_sweep['width'].to_numpy(), spike_in_sweep['peak_t'].to_numpy(), start=param_dict['start'], end=param_dict['end'])[0]
        isi_bin = build_running_bin(np.diff(spike_in_sweep['peak_t'].to_numpy()), spike_in_sweep['peak_t'].to_numpy()[:-1], start=param_dict['start'], end=param_dict['end'])[0]        
        sweep_running_bin = pd.DataFrame(data=np.hstack((trough_averge, peak_average, peak_max_rise, peak_max_down, peak_width, isi_bin)).reshape(1,-1), columns=_run_labels, index=[real_sweep_number])
        spike_train_df = pd.DataFrame(spike_train, index=[0])
        nan_series = pd.DataFrame(np.full(abs(spike_count-1), np.nan))
        #spike_train_df = spike_train_df.append(nan_series)
        spike_in_sweep['spike count'] = np.hstack((spike_count, np.full(abs(spike_count-1), np.nan)))
        spike_in_sweep['sweep Number'] = np.full(abs(spike_count), int(real_sweep_number))
        dict_spike_df["spike_amp" + real_sweep_number + " 1"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[0] - spike_in_sweep['threshold_v'].to_numpy()[0])
        dict_spike_df["spike_thres" + real_sweep_number + " 1"] = spike_in_sweep['threshold_v'].to_numpy()[0]
        dict_spike_df["spike_peak" + real_sweep_number + " 1"] = spike_in_sweep['peak_v'].to_numpy()[0]
        dict_spike_df["spike_rise" + real_sweep_number + " 1"] = spike_in_sweep['upstroke'].to_numpy()[0]
        dict_spike_df["spike_decay" + real_sweep_number + " 1"] = spike_in_sweep['downstroke'].to_numpy()[0]
        dict_spike_df["spike_AHP 1" + real_sweep_number + " "] = spike_in_sweep['fast_trough_v'].to_numpy()[0]
        dict_spike_df["spike_AHP height 1" + real_sweep_number + " "] = abs(spike_in_sweep['peak_v'].to_numpy()[0] - spike_in_sweep['fast_trough_v'].to_numpy()[0])
        dict_spike_df["latency_" + real_sweep_number + " latency"] = spike_train['latency']
        dict_spike_df["spike_width" + real_sweep_number + "1"] = spike_in_sweep['width'].to_numpy()[0]
        curve = exp_growth_factor(abf.sweepX, abf.sweepY, abf.sweepC, int(spike_in_sweep['threshold_index'].to_numpy()[0]))
        dict_spike_df["exp growth tau1" + real_sweep_number] = curve[2]
        dict_spike_df["exp growth tau2" + real_sweep_number] = curve[-1]
                
        if spike_count > 2:
            f_isi = spike_in_sweep['peak_t'].to_numpy()[-1]
            l_isi = spike_in_sweep['peak_t'].to_numpy()[-2]
            dict_spike_df["last_isi" + real_sweep_number + " isi"] = [abs( f_isi- l_isi )]
            spike_in_sweep['isi_'] = np.hstack((np.diff(spike_in_sweep['peak_t'].to_numpy()), np.nan))
            dict_spike_df["min_isi" + real_sweep_number + " isi"] = np.nanmin(np.hstack((np.diff(spike_in_sweep['peak_t'].to_numpy()), np.nan)))
            #dict_spike_df["spike_" + real_sweep_number + " 2"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[1] - spike_in_sweep['threshold_v'].to_numpy()[1])
            #dict_spike_df["spike_" + real_sweep_number + " 3"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[-1] - spike_in_sweep['threshold_v'].to_numpy()[-1])
            #dict_spike_df["spike_" + real_sweep_number + "AHP 2"] = spike_in_sweep['fast_trough_v'].to_numpy()[1]
            #dict_spike_df["spike_" + real_sweep_number + "AHP 3"] = spike_in_sweep['fast_trough_v'].to_numpy()[-1]
            #dict_spike_df["spike_" + real_sweep_number + "AHP height 2"] = abs(spike_in_sweep['peak_v'].to_numpy()[1] - spike_in_sweep['fast_trough_v'].to_numpy()[1])
            #dict_spike_df["spike_" + real_sweep_number + "AHP height 3"] = abs(spike_in_sweep['peak_v'].to_numpy()[-1] - spike_in_sweep['fast_trough_v'].to_numpy()[-1])
            #dict_spike_df["spike_width" + real_sweep_number + "2"] = spike_in_sweep['width'].to_numpy()[1]
            #dict_spike_df["spike_width" + real_sweep_number + "3"] = spike_in_sweep['width'].to_numpy()[-1]
            for label in ipfx_train_feature_labels:
                try:
                    dict_spike_df[label + real_sweep_number] = spike_train[label]
                except:
                    dict_spike_df[label + real_sweep_number] = np.nan
            
        else:
            dict_spike_df["last_isi" + real_sweep_number + " isi"] = [np.nan]
            spike_in_sweep['isi_'] = np.hstack((np.full(abs(spike_count), np.nan)))
            dict_spike_df["min_isi" + real_sweep_number + " isi"] = [spike_train['first_isi']]
            for label in ipfx_train_feature_labels:
                dict_spike_df[label + real_sweep_number] = [np.nan]
        spike_in_sweep = spike_in_sweep.join(spike_train_df)
        print("Processed Sweep " + str(real_sweep_number) + " with " + str(spike_count) + " aps")
        df = df.append(spike_in_sweep, ignore_index=True, sort=True)
    else:
        dict_spike_df["latency_" + real_sweep_number + " latency"] = [np.nan]
        dict_spike_df["first_isi " + real_sweep_number + " isi"] = [np.nan]
        dict_spike_df["last_isi" + real_sweep_number + " isi"] = [np.nan]
        dict_spike_df["spike_amp" + real_sweep_number + " 1"] = [np.nan]
        dict_spike_df["spike_thres" + real_sweep_number + " 1"] = [np.nan]
        dict_spike_df["spike_rise" + real_sweep_number + " 1"] = [np.nan]
        dict_spike_df["spike_decay" + real_sweep_number + " 1"] = [np.nan]
        #dict_spike_df["spike_" + real_sweep_number + " 2"] = [np.nan]
        #dict_spike_df["spike_" + real_sweep_number + " 3"] = [np.nan]
        dict_spike_df["spike_AHP 1" + real_sweep_number + " "] = [np.nan]
        dict_spike_df["spike_peak" + real_sweep_number + " 1"] = [np.nan]
        dict_spike_df["spike_AHP height 1" + real_sweep_number + " "] = [np.nan]
        #dict_spike_df["spike_" + real_sweep_number + "AHP 2"] = [np.nan]
        #dict_spike_df["spike_" + real_sweep_number + "AHP 3"] = [np.nan]
        #dict_spike_df["spike_" + real_sweep_number + "AHP height 2"] = [np.nan]
        #dict_spike_df["spike_" + real_sweep_number + "AHP height 3"] = [np.nan]
        dict_spike_df["latency_" + real_sweep_number + " latency"] = [np.nan]
        dict_spike_df["spike_width" + real_sweep_number + "1"] = [np.nan]
        #dict_spike_df["spike_width" + real_sweep_number + "2"] = [np.nan]
        #dict_spike_df["spike_width" + real_sweep_number + "3"] = [np.nan]
        dict_spike_df["min_isi" + real_sweep_number + " isi"] = [np.nan]
        sweep_running_bin = pd.DataFrame(data=nan_row_run, columns=_run_labels, index=[real_sweep_number])
        dict_spike_df["exp growth tau1" + real_sweep_number] = [np.nan]
        dict_spike_df["exp growth tau2" + real_sweep_number] = [np.nan]
        for label in ipfx_train_feature_labels:
            dict_spike_df[label + real_sweep_number] = [np.nan]
    #dict_spike_df = dict_spike_df.join(spike_train_df)
    sweep_running_bin['Sweep Number'] = [real_sweep_number]
    sweep_running_bin['filename'] = [abf.abfID]
    sweep_running_bin['foldername'] = [os.path.dirname(abf.abfFilePath)]
    sag, taum, voltage = subthres_a(abf.sweepX, abf.sweepY, abf.sweepC, param_dict['start'], param_dict['end'])
    dict_spike_df["Sag Ratio " + real_sweep_number + ""] = sag
    dict_spike_df["Taum " + real_sweep_number + ""] = taum
    temp_running_bin = temp_running_bin.append(sweep_running_bin)
    #append the dict as new columns to the temp_spike_df
    temp_spike_df = temp_spike_df.assign(**dict_spike_df)
    return temp_spike_df, df, temp_running_bin


def _build_full_df(abf, temp_spike_df, df, temp_running_bin, sweepList):
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
                
def compute_current_injection_features(sweepX, sweepY, sweepC):
    """_summary_
    Args:
        sweepX (_type_): A 2d array of time values for each sweep in the format [sweep, time]
        sweepY (_type_): A 2d array of voltage values for each sweep in the format [sweep, voltage]
        sweepC (_type_): A 2d array of current values for each sweep in the format [sweep, current]

    returns:
        current_injection_features (dict): A dictionary of current injection features for each sweep
    """
    
    current_injection_features = {}

    for sweepN, sweep in enumerate(sweepC):
        #get the unique current injections
        unique_current_injections = np.unique(sweep)
        #figure out how many current injections there are, if there are more than 6, we will only take the first 3 and last 3
        if len(unique_current_injections) > 6:
            unique_current_injections = np.hstack((unique_current_injections[:3], unique_current_injections[-3:]))
        #append them in the order they appear in the sweepC array
        if np.any(unique_current_injections<0):
            negative_current_injections = unique_current_injections[unique_current_injections<0]
            for i, current_injection in enumerate(negative_current_injections):
                current_injection_features[f"sweep_{sweepN}_hyperpolarizing_{str(i)}_current"] = current_injection
        if np.any(unique_current_injections>0.0):
            positive_current_injections = unique_current_injections[unique_current_injections>0]
            for i, current_injection in enumerate(positive_current_injections):
                current_injection_features[f"sweep_{sweepN}_{str(i)}_current"] = current_injection
        else:
            for i, current_injection in enumerate(unique_current_injections):
                current_injection_features[f"sweep_{sweepN}_{str(i)}_current"] = current_injection
    
    
    #now compute some other features
    if len(sweepC) > 1:
        #compute the difference between the first and second current injection, this is the delta current
        current_injection_features["current_delta"] = current_injection_features[f"sweep_{0}_{0}_current"] - current_injection_features[f"sweep_{1}_{0}_current"]
    
    #merge the hyperpolarizing current injections into a single feature, if they are all the same
    if len(current_injection_features) > 1:
        hyperpolarizing_current_injections = [current_injection_features[key] for key in current_injection_features.keys() if "hyperpolarizing" in key]
        if len(np.unique(hyperpolarizing_current_injections)) == 1:
            current_injection_features["hyperpolarizing_current"] = hyperpolarizing_current_injections[0]
    
    return current_injection_features

