#############
# this script is built to do some additional processing on the output of ipfx
#  we will be computing some features needed / desired for the inoue lab
# here we will be wrangling some of the data frames and saving them to csv files
# we will also be generating some plots and saving them to the same folder
#############
import numpy as np
import os
import pandas as pd

from .patch_utils import *
from .patch_subthres import *

ipfx_train_feature_labels =['adapt',  'isi_cv', 'mean_isi', 'median_isi', 
       'avg_rate']
#'first_isi', 'latency',
running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width', 'isi']

subsheets_spike = {'full sheet': ['']}
#old subsheets 'spike count':['spike count'], 'rheobase features':['rheobase'], 
    #                'mean':['mean'], 'isi':['isi'], 'latency': ['latency_'], 'current':['current'],'QC':['QC'], 
        #            'spike features':['spike_'], 'subthres features':['baseline voltage', 'Sag', 'Taum'], 



def save_data_frames(dfs, df_spike_count, df_running_avg_count, root_fold='', tag='', savespikeFinder=True, saveRunningAvg=True, saveRaw=False):
    dfs, df_spike_count, df_running_avg_count = organize_data_frames(dfs, df_spike_count, df_running_avg_count)
    if savespikeFinder:
        with pd.ExcelWriter(root_fold + '/spike_count_' + tag + '.xlsx') as runf:
            cols = df_spike_count.columns.values
            df_ind = df_select_by_col(df_spike_count, ['foldername', 'filename'])
            df_ind = df_ind.loc[:,['foldername', 'filename']]
            index = pd.MultiIndex.from_frame(df_ind)
            for key, p in subsheets_spike.items():
                temp_ind = df_select_by_col(df_spike_count, p)
                temp_df = temp_ind.set_index(index)
                temp_df.to_excel(runf, sheet_name=key)
            if saveRaw:
                dfs.to_excel(runf, sheet_name="RAW")
    if saveRunningAvg:
        with pd.ExcelWriter(root_fold + '/running_avg_' + tag + '.xlsx') as runf:
            cols = df_running_avg_count.columns.values
            df_ind = df_running_avg_count.loc[:,['foldername', 'filename', 'Sweep Number']]
            index = pd.MultiIndex.from_frame(df_ind)
            for p in running_lab:
                temp_ind = [p in col for col in cols]
                temp_df = df_running_avg_count.set_index(index).loc[:,temp_ind]
                temp_df.to_excel(runf, sheet_name=p)
    print("data frames saved to excel")

def organize_data_frames(dfs, df_spike_count, df_running_avg_count):
    #here we will reorder the columns,
    #we want to put the file name and folder name first
    cols = df_spike_count.columns.values
    #get the index of the file name and folder name
    ind = [col in ['filename', 'foldername', 'protocol'] for col in cols]
    
    #now we want the mean and rheobase features
    mean_ind = [np.logical_and('mean' in col,'mean_isi0' not in col) for col in cols ]

    rheo_ind = ['rheobase' in col for col in cols]
    #now get the spike_count features
    spike_ind = ['spike count' in col for col in cols]
    #now the stimuli features
    stim_ind = ['stimuli_length' in col for col in cols]
    sample_rate = ['sample_rate' in col for col in cols]
    epoch_ind = ['epoch' in col for col in cols]
    #now the rest of the columns are alphabetical
    #now we want to sort the columns
    cols_sort = np.hstack((cols[ind], cols[mean_ind], cols[rheo_ind], cols[spike_ind], cols[stim_ind], cols[epoch_ind]))
    colother = np.setdiff1d(cols, cols_sort)
    cols_sort = np.hstack((cols_sort, np.sort(colother)))
    assert len(cols_sort) == len(cols)
    #now we want to sort the rows
    df_spike_count = df_spike_count[cols_sort]
    return dfs, df_spike_count, df_running_avg_count






# TODO <--- this is a mess, clean it up
# Ensure functions do no require the abf object
# functions should not depend on further analysis, only concat of dataframes
# functions should not depend on the order of the sweeps
def _build_sweepwise_dataframe(real_sweep_number, spike_in_sweep, spike_train, temp_spike_df, df, temp_running_bin, param_dict):
    """Build a "sweepwise" dataframe for a single sweep. Essentialy compacts the features of a single sweep into several singluar columns
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
    
    #pregenerate the running bin labels
    time_bins = np.arange(param_dict['start']*1000, param_dict['end']*1000+20, 20)
    _run_labels = []
    for p in running_lab:
            temp_lab = []
            for x in time_bins :
                temp_lab = np.hstack((temp_lab, f'{p} {x} bin AVG'))
            _run_labels.append(temp_lab)
    _run_labels = np.hstack(_run_labels).tolist()
    nan_row_run = np.ravel(np.full((len(running_lab), time_bins.shape[0]), np.nan)).reshape(1,-1)


    if spike_count > 0:
        
        # Calculate running averages
        trough_average = build_running_bin(spike_in_sweep['fast_trough_v'], spike_in_sweep['peak_t'], start=param_dict['start'], end=param_dict['end'])[0]
        peak_average = build_running_bin(spike_in_sweep['peak_v'], spike_in_sweep['peak_t'], start=param_dict['start'], end=param_dict['end'])[0]
        peak_max_rise = build_running_bin(spike_in_sweep['upstroke'], spike_in_sweep['peak_t'], start=param_dict['start'], end=param_dict['end'])[0]
        peak_max_down = build_running_bin(spike_in_sweep['downstroke'], spike_in_sweep['peak_t'], start=param_dict['start'], end=param_dict['end'])[0]
        peak_width = build_running_bin(spike_in_sweep['width'], spike_in_sweep['peak_t'], start=param_dict['start'], end=param_dict['end'])[0]
        isi_bin = build_running_bin(np.diff(spike_in_sweep['peak_t']), spike_in_sweep['peak_t'][:-1], start=param_dict['start'], end=param_dict['end'])[0]

        # Create dataframes of the running averages
        sweep_running_bin = pd.DataFrame(data=np.hstack((trough_average, peak_average, peak_max_rise, peak_max_down, peak_width, isi_bin)).reshape(1,-1), columns=_run_labels, index=[real_sweep_number])
        spike_train_df = pd.DataFrame(spike_train, index=[0])
        
        spike_in_sweep['spike count'] = np.hstack((spike_count, np.full(abs(spike_count-1), np.nan)))
        spike_in_sweep['sweep Number'] = np.full(abs(spike_count), int(real_sweep_number))

        dict_spike_df["first_isi " + real_sweep_number + " isi"] = [spike_train['first_isi']]
        dict_spike_df["spike_amp" + real_sweep_number + " 1"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[0] - spike_in_sweep['threshold_v'].to_numpy()[0])
        dict_spike_df["spike_thres" + real_sweep_number + " 1"] = spike_in_sweep['threshold_v'].to_numpy()[0]
        dict_spike_df["spike_peak" + real_sweep_number + " 1"] = spike_in_sweep['peak_v'].to_numpy()[0]
        dict_spike_df["spike_rise" + real_sweep_number + " 1"] = spike_in_sweep['upstroke'].to_numpy()[0]
        dict_spike_df["spike_decay" + real_sweep_number + " 1"] = spike_in_sweep['downstroke'].to_numpy()[0]
        dict_spike_df["spike_AHP 1" + real_sweep_number + " "] = spike_in_sweep['fast_trough_v'].to_numpy()[0]
        dict_spike_df["spike_AHP height 1" + real_sweep_number + " "] = abs(spike_in_sweep['peak_v'].to_numpy()[0] - spike_in_sweep['fast_trough_v'].to_numpy()[0])
        dict_spike_df["latency_" + real_sweep_number + " latency"] = spike_train['latency']
        dict_spike_df["spike_width" + real_sweep_number + "1"] = spike_in_sweep['width'].to_numpy()[0]
        
                
        if spike_count >= 2:
            f_isi = spike_in_sweep['peak_t'].to_numpy()[-1]
            l_isi = spike_in_sweep['peak_t'].to_numpy()[-2]
            dict_spike_df["last_isi" + real_sweep_number + " isi"] = [abs( f_isi- l_isi )]
            spike_in_sweep['isi_'] = np.hstack((np.diff(spike_in_sweep['peak_t'].to_numpy()), np.nan))
            dict_spike_df["min_isi" + real_sweep_number + " isi"] = np.nanmin(np.hstack((np.diff(spike_in_sweep['peak_t'].to_numpy()), np.nan)))
            # dict_spike_df["spike_" + real_sweep_number + " 2"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[1] - spike_in_sweep['threshold_v'].to_numpy()[1])
            # dict_spike_df["spike_" + real_sweep_number + " 3"] = np.abs(spike_in_sweep['peak_v'].to_numpy()[-1] - spike_in_sweep['threshold_v'].to_numpy()[-1])
            # dict_spike_df["spike_" + real_sweep_number + "AHP 2"] = spike_in_sweep['fast_trough_v'].to_numpy()[1]
            # dict_spike_df["spike_" + real_sweep_number + "AHP 3"] = spike_in_sweep['fast_trough_v'].to_numpy()[-1]
            # dict_spike_df["spike_" + real_sweep_number + "AHP height 2"] = abs(spike_in_sweep['peak_v'].to_numpy()[1] - spike_in_sweep['fast_trough_v'].to_numpy()[1])
            # dict_spike_df["spike_" + real_sweep_number + "AHP height 3"] = abs(spike_in_sweep['peak_v'].to_numpy()[-1] - spike_in_sweep['fast_trough_v'].to_numpy()[-1])
            # dict_spike_df["spike_width" + real_sweep_number + "2"] = spike_in_sweep['width'].to_numpy()[1]
            # dict_spike_df["spike_width" + real_sweep_number + "3"] = spike_in_sweep['width'].to_numpy()[-1]
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
        # dict_spike_df["spike_" + real_sweep_number + " 2"] = [np.nan]
        # dict_spike_df["spike_" + real_sweep_number + " 3"] = [np.nan]
        dict_spike_df["spike_AHP 1" + real_sweep_number + " "] = [np.nan]
        dict_spike_df["spike_peak" + real_sweep_number + " 1"] = [np.nan]
        dict_spike_df["spike_AHP height 1" + real_sweep_number + " "] = [np.nan]
        # dict_spike_df["spike_" + real_sweep_number + "AHP 2"] = [np.nan]
        # dict_spike_df["spike_" + real_sweep_number + "AHP 3"] = [np.nan]
        # dict_spike_df["spike_" + real_sweep_number + "AHP height 2"] = [np.nan]
        # dict_spike_df["spike_" + real_sweep_number + "AHP height 3"] = [np.nan]
        dict_spike_df["latency_" + real_sweep_number + " latency"] = [np.nan]
        dict_spike_df["spike_width" + real_sweep_number + "1"] = [np.nan]
        # dict_spike_df["spike_width" + real_sweep_number + "2"] = [np.nan]
        # dict_spike_df["spike_width" + real_sweep_number + "3"] = [np.nan]
        dict_spike_df["min_isi" + real_sweep_number + " isi"] = [np.nan]
        sweep_running_bin = pd.DataFrame(data=nan_row_run, columns=_run_labels, index=[real_sweep_number])
        for label in ipfx_train_feature_labels:
            dict_spike_df[label + real_sweep_number] = [np.nan]

    
    sweep_running_bin['Sweep Number'] = [real_sweep_number]
    #sweep_running_bin['filename'] = [abf.abfID]
    #sweep_running_bin['foldername'] = [os.path.dirname(abf.abfFilePath)]
  
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
                

