# This script applies EPSC thresholding to the time series data
import numpy as np
import scipy.stats as stats

def epsc_threshold(data, thres=-5, rearm=None):
    """
    detect ESPCs via thresholding

    return thresholded
    """
    #get points where data crosses the threshold
    thresholded = np.where(data < thres, data, 0)
    #count the first instance
    if rearm is not None:
        thresholded = np.where(thresholded > 0, thresholded, np.nan)
        #count the first instance
        first_instance = np.nanargmin(thresholded) if np.any(np.isnan(thresholded)) else -1
        return thresholded, first_instance
    

    return thresholded