import numpy as np
import matplotlib.pyplot as plt
import os
import pyabf
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.signal as signal

def load_protocols(path):
    protocol = []
    for root,dir,fileList in os.walk(path):
        for filename in fileList:
            if filename.endswith(".abf"):
                try:
                    file_path = os.path.join(root,filename)
                    abf = pyabf.ABF(file_path, loadData=False)
                    protocol = np.hstack((protocol, abf.protocol))
                except:
                    print('error processing file ' + file_path)
    return np.unique(protocol)


def plotabf(abf, spiketimes, lowerlim, upperlim, sweep_plots):
   try:
    if sweep_plots[0] == -1:
        pass
    else:
        plt.figure(num=2, figsize=(16,6))
        plt.clf()
        cm = plt.get_cmap("Set1") #Changes colour based on sweep number
        if sweep_plots[0] == 0:
            sweepList = abf.sweepList
        else:
            sweepList = sweep_plots - 1
        colors = [cm(x/np.asarray(sweepList).shape[0]) for x,_ in enumerate(sweepList)]
        
        plt.autoscale(True)
        plt.grid(alpha=0)

        plt.xlabel(abf.sweepLabelX)
        plt.ylabel(abf.sweepLabelY)
        plt.title(abf.abfID)

        for c, sweepNumber in enumerate(sweepList):
            abf.setSweep(sweepNumber)
            
            spike_in_sweep = (spiketimes[spiketimes[:,1]==int(sweepNumber+1)])[:,0]
            i1, i2 = int(abf.dataRate * lowerlim), int(abf.dataRate * upperlim) # plot part of the sweep
            dataX = abf.sweepX
            dataY = abf.sweepY
            colour = colors[c]
            sweepname = 'Sweep ' + str(sweepNumber)
            plt.plot(dataX, dataY, color=colour, alpha=1, lw=1, label=sweepname)
            
            plt.scatter(dataX[spike_in_sweep[:]], dataY[spike_in_sweep[:]], color=colour, marker='x')
           
        

        plt.xlim(abf.sweepX[i1], abf.sweepX[i2])
        plt.legend()
        
        plt.savefig(abf.abfID +'.png', dpi=600)
        plt.pause(0.05)
   except:
        print('plot failed')

def build_running_bin(array, time, start, end, bin=20, time_units='s', kind='nearest'):
    if time_units == 's':
        start = start * 1000
        end = end* 1000

        time = time*1000
    time_bins = np.arange(start, end+bin, bin)
    binned_ = np.full(time_bins.shape[0], np.nan, dtype=np.float64)
    index_ = np.digitize(time, time_bins)
    uni_index_ = np.unique(index_)
    for time_ind in uni_index_:
        data = np.asarray(array[index_==time_ind])
        data = np.nanmean(data)
        binned_[time_ind] = data
    nans = np.isnan(binned_)
    if np.any(nans):
        if time.shape[0] > 1:
            f = interpolate.interp1d(time, array, kind=kind, fill_value="extrapolate")
            new_data = f(time_bins)
            binned_[nans] = new_data[nans]
        else:
            binned_[nans] = np.nanmean(array)
    return binned_, time_bins

def create_dir(fp):
    if os.path.exists(fp):
        pass
    else:
        os.makedirs(fp)
    return fp


def df_select_by_col(df, string_to_find):
    columns = df.columns.values
    out = []
    for col in columns:
        string_found = [x in col for x in string_to_find]
        if np.any(string_found):
            out.append(col)
    return df[out]

def time_to_idx(dataX, time):
    if dataX.nDim > 1:
        dataX = dataX[0, :]
    
    idx = np.argmin(np.abs(time-dataX))
    return idx

def idx_to_time(dataX, idx):
    pass

def find_stim_changes(dataI):
    diff_I = np.diff(dataI)
    infl = np.nonzero(diff_I)[0]
    
    return infl

def find_downward(dataI):
    diff_I = np.diff(dataI)
    downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
    return downwardinfl

def find_non_zero_range(dataT, dataI):
    non_zero_points = np.nonzero(dataI)[0]
    return (dataT[non_zero_points[0]], dataT[non_zero_points[-1]])

def filter_bessel(data_V, fs, cutoff):
    """_summary_

    Args:
        data_V (_type_): _description_
        abf (_type_): _description_
        cutoff (_type_): _description_

    Returns:
        _type_: _description_
    """
    #filter the abf with 5 khz lowpass
    #if the cutoff is lower than critical frequency, filter the data
    try:
        b, a = signal.bessel(4, cutoff, 'low', norm='phase', fs=fs)
        dataV = signal.filtfilt(b, a, data_V)
    except:
        dataV = data_V
    return dataV



