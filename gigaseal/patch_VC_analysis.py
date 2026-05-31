import sys
import numpy as np
from numpy import genfromtxt
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.stats
from ipfx import subthresh_features as subt
from patch_utils import find_stim_changes, time_to_idx
import pyabf

def vc_sweepwise_linregress(dataX, dataY, dataC, tstart=None, tend=None):
    if tstart is None:
        stim_ = find_stim_changes(dataC)
        idx_on = stim[0]
    else:
        idx_on = time_to_idx(dataX, start)
    
    baseline_I = np.nanmean(dataY[:, :idx_on], axis=1)
    resp_I = np.amax(dataY[:, idx_on:], axis=1)
    holding_v = np.nanmean(dataC[:, :idx_on])
    res = stats.linregress()