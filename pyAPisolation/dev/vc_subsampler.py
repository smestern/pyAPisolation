# %%
import pyabf
import numpy as np
from scipy.signal import savgol_filter, bessel, filtfilt
import matplotlib.pyplot as plt
import sys
import os
import glob
from pyabf.abfWriter import writeABF1

import pandas as pd
from scipy import stats

def subsample_file(abf=None):
    abf = pyabf.ABF(abf)

    # %%
    # Get the sweep data
    sweep = 0
    abf.setSweep(sweep, channel=0)
    data = abf.sweepY
    stim = abf.sweepC
    time = abf.sweepX
    sr = abf.dataRate

    #baseline correction via linear regression
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(time, data)
    data = data - (slope*time + intercept)

    # apply a low pass filter to 1hz
    # design the filter
    b, a = bessel(4, 1, 'low', fs=sr)
    # apply the filter
    data_base = filtfilt(b, a, data)
    # find periods where the data is 3 standard deviations above or below the mean in the filtered data
    mean = 0.0
    thresh = 5
    idx_fail = np.where(np.logical_or(data_base > mean + thresh, data_base < mean - thresh))[0]



    data_adjusted = data.copy()
    data_adjusted[idx_fail] = 0
    

    # %%
    #now get the full run_time  of the data
    run_time = time[-1]
    subsample_window = 10*sr # 10 seconds
    subsample_step = 60*sr # 5 seconds
    subsampled_data = []
    subsampled_time = []
    for i in range(0, len(data), subsample_step):
        subsampled_data.append(data_adjusted[i:i+subsample_window])
        subsampled_time.append(time[i:i+subsample_window])



    def make_or_get_fold(fold):
        if not os.path.exists(fold):
            os.makedirs(fold)
        return fold


    base_name = os.path.basename(abf.abfFilePath).split('.')[0]
    dir_name = make_or_get_fold(os.path.join(os.path.dirname(abf.abfFilePath), f"{base_name}_subsampled"))
    data_out = {}
    for d, t in zip(subsampled_data, subsampled_time):
        writeABF1(filename=f"{dir_name}/{base_name}_{t[0]:.0f}.abf", sweepData=d.reshape(-1, 1), units='pA', sampleRateHz=sr)
        data_out[f"{base_name}_{t[0]:.0f}.abf"] = {'time_start': t[0], 'time_end': t[-1]}

   
    df = pd.DataFrame(data_out).T
    df.to_csv(f"{dir_name}/{base_name}_data.csv")



if __name__ == "__main__":
    FOLDER = os.path.expanduser("~/dropbox/20/")
    PROTOCOL = 'Gap free'
    for abf_file in glob.glob(f"{FOLDER}/*.abf"):
        abf = pyabf.ABF(abf_file)
        if abf.protocol == PROTOCOL:
            print(f"Subsampling {abf_file}")
            subsample_file(abf_file)
