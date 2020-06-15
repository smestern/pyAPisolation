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
from abf_utils import *
from abf_subthres import *



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



