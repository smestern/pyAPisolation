import numpy as np
import os
import glob
import pandas as pd
from scipy import stats

def find_zero(realC):
    #expects 1d array
    zero_ind = np.where(realC == 0)[0]
    return zero_ind

def find_baseline(zero_ind):
    #the baseline will be the first continious set of zeros
    baseline_idx = np.where(np.diff(zero_ind) > 1)[0]
    if len(baseline_idx) == 0:
        baseline_idx = len(zero_ind)
    else:
        baseline_idx = baseline_idx[0]
    return zero_ind[0:baseline_idx+1]

def compute_vm_drift(realY, zero_ind):
    sweep_wise_mean = np.mean(realY[:,zero_ind], axis=1)
    mean_drift = np.abs(np.amax(sweep_wise_mean) - np.amin(sweep_wise_mean))
    abs_drift = np.abs(np.amax(realY[:,zero_ind]) - np.amin(realY[:,zero_ind]))
   
    return mean_drift, abs_drift


def compute_rms(realY, zero_ind):
    mean = np.mean(realY[:,zero_ind], axis=1)
    rms = []
    for x in np.arange(mean.shape[0]):
        temp = np.sqrt(np.mean(np.square(realY[x,zero_ind] - mean[x])))
        rms = np.hstack((rms, temp))
    full_mean = np.mean(rms)
    return full_mean, np.amax(rms)

def run_qc(realY, realC):
    zero_ind = find_zero(realC[0,:])
    zero_ind = find_baseline(zero_ind)
    mean_rms, max_rms = compute_rms(realY, zero_ind)
    mean_drift, max_drift = compute_vm_drift(realY, zero_ind)
    return [mean_rms, max_rms, mean_drift, max_drift]

