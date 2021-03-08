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
from abf_ipfx_dataframes import *

def folder_feature_extract(files, param_dict, plot_sweeps=-1, protocol_name='IC1'):
    debugplot = 0
    running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width']
    dfs = pd.DataFrame()
    df_spike_count = pd.DataFrame()
    df_running_avg_count = pd.DataFrame()
    for root,dir_,fileList in os.walk(files):
        for filename in fileList:
            if filename.endswith(".abf"):
                file_path = os.path.join(root,filename)
                try:
                    abf = pyabf.ABF(file_path)
                
                    if abf.sweepLabelY != 'Clamp Current (pA)' and protocol_name in abf.protocol:
                        print(filename + ' import')
                        temp_spike_df, df, temp_running_bin = analyze_abf(abf, sweeplist=None, plot=plot_sweeps, param_dict=param_dict)
                        df_running_avg_count = df_running_avg_count.append(temp_running_bin)
                        df_spike_count = df_spike_count.append(temp_spike_df, sort=True)
                        dfs = dfs.append(df, sort=True)
                    else:
                        print('Not correct protocol: ' + abf.protocol)
                except:

                    print('Issue Processing ' + filename)
    return dfs, df_spike_count, df_running_avg_count


def save_data_frames(dfs, df_spike_count, df_running_avg_count, root_fold='', tag=''):
    try:
        ids = dfs['__file_name'].unique()
        tempframe = dfs.groupby('__file_name').mean().reset_index()
        tempframe.to_csv(root_fold + '/allAVG_' + tag + '.csv')
        tempframe = dfs.drop_duplicates(subset='__file_name')
        tempframe.to_csv(root_fold + '/allRheo_' + tag + '.csv')
        df_spike_count.to_csv(root_fold + '/spike_count_' + tag + '.csv')
        dfs.to_csv(root_fold + '/allfeatures_' + tag + '.csv')
        with pd.ExcelWriter(root_fold + '/running_avg_' + tag + '.xlsx') as runf:
            cols = df_running_avg_count.columns.values
            df_ind = df_running_avg_count.loc[:,cols[[-1,-2,-3]]]
            index = pd.MultiIndex.from_frame(df_ind)
            for p in running_lab:
                temp_ind = [p in col for col in cols]
                temp_df = df_running_avg_count.set_index(index).loc[:,temp_ind]
                temp_df.to_excel(runf, sheet_name=p)
    except: 
        print('error saving')

class abfFeatExtractor(object):
    """ """
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



