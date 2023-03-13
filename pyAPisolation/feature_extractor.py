import glob
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf
import copy
import multiprocessing as mp
from ipfx import feature_extractor
from ipfx import subthresh_features as subt
import scipy.signal as signal

from .ipfx_df import _build_full_df, _build_sweepwise_dataframe, save_data_frames
from .loadABF import loadABF
from .patch_utils import plotabf, load_protocols, find_non_zero_range, filter_abf
from .patch_subthres import *
from .QC import run_qc
print("feature extractor loaded")
parallel = True
default_dict = {'start': 0, 'end': 0, 'filter': 0, 'stim_find': True}

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
    debugplot = 0
    running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width']
    dfs = pd.DataFrame()
    df_spike_count = pd.DataFrame()
    df_running_avg_count = pd.DataFrame()
    filelist = glob.glob(files + "/**/*.abf", recursive=True)
    spike_count = []
    df_full = []
    df_running_avg = []
    if parallel:
        pool = mp.Pool()
        results = [pool.apply(preprocess_abf, args=(file, param_dict, plot_sweeps, protocol_name)) for file in filelist]
        pool.close()
        ##split out the results
        for result in results:
            temp_res = result
            df_full.append(temp_res[1])
            df_running_avg.append(temp_res[2])
            spike_count.append(temp_res[0])
        pool.join()
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

def analyze_spike_sweep(abf, sweepNumber, param_dict, bessel_filter=None):
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
    spike_in_sweep = spikext.process(dataT, dataV, dataI)
    spike_train = spiketxt.process(dataT, dataV, dataI, spike_in_sweep)
    return spike_in_sweep, spike_train


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

    #iterate through the sweeps
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
        spike_in_sweep, spike_train = analyze_spike_sweep(abf, sweepNumber, param_dict, bessel_filter=bessel_filter) ### Returns the default Dataframe Returned by 
        temp_spike_df, df, temp_running_bin = _build_sweepwise_dataframe(abf, real_sweep_number, spike_in_sweep, spike_train, temp_spike_df, df, temp_running_bin, param_dict)
    temp_spike_df, df, temp_running_bin = _build_full_df(abf, temp_spike_df, df, temp_running_bin, sweepcount)
    x, y ,c = loadABF(abf.abfFilePath)
    #try qc or just return the dataframe
    try:
        _qc_data = run_qc(y, c)
        temp_spike_df['QC Mean RMS'] = _qc_data[0]
        temp_spike_df['QC Mean Sweep Drift'] = _qc_data[2]
    except:
        temp_spike_df['QC Mean RMS'] = np.nan
        temp_spike_df['QC Mean Sweep Drift'] = np.nan
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

            df, avg = analyze_subthres(abf, sweeplist=None,  **param_dict)
            return df, avg
        else:
            print('Not correct protocol: ' + abf.protocol)
            return pd.DataFrame(), pd.DataFrame()
    except:
       return pd.DataFrame(), pd.DataFrame()

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
    




class abfFeatExtractor(object):
    """TODO """
    def __init__(self, abf, start=None, end=None, filter=10.,
                 dv_cutoff=20., max_interval=0.005, min_height=2., min_peak=-30.,
                 thresh_frac=0.05, reject_at_stim_start_interval=0):
        """Initialize SweepFeatures object.-
        Parameters
        ----------
        t : ndarray of times (seconds)
        v : ndarray of voltages (mV)
        i : ndarray of currents (pA)
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