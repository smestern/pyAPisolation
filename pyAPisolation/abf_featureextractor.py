import glob
import os
import sys
import copy
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf

from ipfx import feature_extractor
from ipfx import subthresh_features as subt
print("feature extractor loaded")

from .abf_ipfx_dataframes import _build_full_df, _build_sweepwise_dataframe, save_data_frames
from .loadABF import loadABF
from .patch_utils import plotabf, load_protocols, find_non_zero_range
from .QC import run_qc

default_dict = {'start': 0, 'end': 0, 'filter': 0}

def folder_feature_extract(files, param_dict, plot_sweeps=-1, protocol_name='IC1', para=1):

    debugplot = 0
    running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width']
    dfs = pd.DataFrame()
    df_spike_count = pd.DataFrame()
    df_running_avg_count = pd.DataFrame()
    filelist = glob.glob(files + "/**/*.abf", recursive=True)
    temp_df_spike_count = Parallel(n_jobs= para)(delayed(preprocess_abf)(f, copy.deepcopy(param_dict), plot_sweeps, protocol_name) for f in filelist)
    df_spike_count = pd.concat(temp_df_spike_count, sort=True)
    
     
    return dfs, df_spike_count, df_running_avg_count

def preprocess_abf(file_path, param_dict, plot_sweeps, protocol_name):
    
    try:
        abf = pyabf.ABF(file_path)
                    
        if abf.sweepLabelY != 'Clamp Current (pA)' and protocol_name in abf.protocol:
            print(file_path + ' import')
            temp_spike_df, df, temp_running_bin = analyze_abf(abf, sweeplist=None, plot=plot_sweeps, param_dict=param_dict)
            return temp_spike_df
        else:
            print('Not correct protocol: ' + abf.protocol)
            return pd.DataFrame()
    except:
        return pd.DataFrame()

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
        stim_find = param_dict.pop('stim_find')
        #for now if user wants to filter by stim time we will just use the first sweep
        if stim_find:
            abf.setSweep(abf.sweepList[-1])
            start, end = find_non_zero_range(abf.sweepX, abf.sweepC)
            param_dict['end'] = end
            param_dict['start'] = start
            print('Stimulation time found: ' + str(start) + ' to ' + str(end))

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
            spike_in_sweep, spike_train = analyze_spike_sweep(abf, sweepNumber, param_dict) ### Returns the default Dataframe Returned by 
            temp_spike_df, df, temp_running_bin = _build_sweepwise_dataframe(abf, real_sweep_number, spike_in_sweep, spike_train, temp_spike_df, df, temp_running_bin, param_dict)
        temp_spike_df, df, temp_running_bin = _build_full_df(abf, temp_spike_df, df, temp_running_bin, sweepcount)
        x, y ,c = loadABF(abf.abfFilePath)
        _qc_data = run_qc(y, c)
        temp_spike_df['QC Mean RMS'] = _qc_data[0]
        temp_spike_df['QC Mean Sweep Drift'] = _qc_data[2]
        try:
            spiketimes = np.transpose(np.vstack((np.ravel(df['peak_index'].to_numpy()), np.ravel(df['sweep Number'].to_numpy()))))
            plotabf(abf, spiketimes, param_dict['start'], param_dict['end'], plot)
        except:
            pass
        return temp_spike_df, df, temp_running_bin







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



