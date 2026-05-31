
import numpy as np
import matplotlib.pyplot as plt
import pyabf
import scipy.signal as signal
from scipy.fftpack import fft, fftfreq
import scipy.stats as stats
from sklearn.preprocessing import maxabs_scale
from scipy.optimize import curve_fit
from scipy.stats  import mode

import scipy.io as sio
import neo
import quantities as pq
from elephant import statistics
from elephant import kernels
from ipfx.feature_extractor import SpikeFeatureExtractor
import pandas as pd
import glob


PA_SCALE = 1e-12
MV_SCALE = 1e-3
GOHM = 1000000000

def compute_impedence_freq(x, y, c, f_range, f_step=100, plot=False):
    

    dt = x[0, 1] - x[0,0]
    f_x = fftfreq(x.shape[1], dt)
    f_y = fft(y * pq.mV, axis=1)
    f_c = fft(c * pq.pA, axis=1)
    f_y = f_y[:, f_x > 0]
    f_c = f_c[:, f_x > 0]
    f_x = f_x[f_x > 0]

    f_y = f_y[:, f_x < f_range[1]]
    f_c = f_c[:, f_x < f_range[1]]
    f_x = f_x[f_x < f_range[1]]

    N =  len(f_y)
    impedence_response = np.abs(f_y / f_c) / GOHM
    if plot:
        plt.clf()
        plt.figure(num=90)
        plt.plot(f_x, np.mean(impedence_response, axis=0))
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Impedence Response')
        plt.pause(5)
    return f_x, impedence_response

def exp_decay(t, a, tau, c):
    return a * np.exp(-tau * t) + c

def equal_array_size_from_list(list_):
    """Takes a list of arrays and returns a list of arrays with the same size

    Args:
        list_ (list): list of arrays

    Returns:
        list: list of arrays with the same size
    """
    max_size = np.max([len(x) for x in list_])
    return np.array([np.pad(x, (0, max_size - len(x)), 'constant', constant_values=np.nan) for x in list_])

def equal_dict_from_list_of_dicts(list_):
    #find the dict with the most keys
    max_keys = np.argmax([len(x.keys()) for x in list_])
    #get the keys
    keys = list_[max_keys].keys()
    #create a dict with the same keys
    updated_dict_list = []
    for dict in list_:
        updated_dict = {}
        for key in keys:
            if key not in dict.keys():
                updated_dict.update({key:np.nan})
            else:
                updated_dict.update({key:dict[key]})
        updated_dict_list.append(updated_dict)
    return updated_dict_list


def compute_real_sweep_num(num):
    #frontfill the sweep number with zeros so it is 4 digits long
    return str(num).zfill(4)

def rmp_mode(dataV, dataC, round_factor=10):
    """Compute the resting membrane potential using the mode of the voltage trace before the stimulus. Using a rounding factor to avoid floating point issues.
    Args:
        dataV (np.array): the voltage data
        dataC (np.array): the current data
        round_factor (int, optional): the rounding factor to use. Defaults to 10 (0.1mV resolution). set to 1 for 1mV resolution.
    Returns:
        float: the resting membrane potential
    """
    #take upto the first non zero
    pre = np.where(dataC[0]>0)[0][0]
    if pre==0:
        #skip the nonzeros at the begining 
        pre = np.where(np.round(dataC[0])>0)[0][0]
    mode_vm = mode(np.ravel(np.round(dataV[:, :pre]*round_factor)/round_factor), axis=None, nan_policy='omit')[0]
    if not np.isscalar(mode_vm):
        mode_vm = mode_vm[0] #depending on the version of scipy, mode returns an array or a single value (here we make sure it is a single value)
    return mode_vm

def determine_protocol_to_use(abf):
    """
    pyabf has trouble finding the protocol, so we need to manually determine it. This function finds the appropriate protocol to use
    Args:
        abf (str): the path to the abf file
    returns:
        protocol (str): the protocol to use
    """
    abf = pyabf.ABF(abf, loadData=False)
    protocol = abf._stringsIndexed.lDACFilePath[0]
    #get the basename of the protocol
    protocol = os.path.basename(protocol)
    #look for this abf file in the cwd
    if os.path.exists(protocol):
        #if it exists, load it
        _ = loadABF(protocol)
        #get the protocol
        return os.path.abspath(protocol)
    else:
        return None
    
def run_analysis_abf(abf, stim_abf, line_baseline=True, compute_inst_freq=True, mask_spikes=True, plot=False, mask_sweeps=False):
    """Run EPSP analysis over abf files. In this case, since we are doing unpaired analysis, the stim and response are passed in as separate files.
    Args:
        abf (str): path to the abf file containing the voltage response
        stim_abf (str): path to the abf file containing the stimulus
        line_baseline (bool, optional): Applies linear regression to baseline the sweep,
                                        accounting for voltage drift. Defaults to True.
        compute_inst_freq (bool, optional): Compute the inst. freq. response for each sweep. 
                                            Takes a long time and can be disabled. Defaults to True.
        mask_spikes (bool, optional): Masks the spikes in the response. Defaults to True.
        plot (bool, optional): plot the gathered data. Defaults to False.
        mask_sweeps (bool, optional): Masks the sweeps that have a variance ratio greater than 1.5. Defaults to False.
    Returns:
        dict: containing the resp means and FFT data
    """
    
    #internal imports
    from ..loadFile import loadABF
    #Load the EPSP DATA
    f_x, f_y, f_c, obj2 = loadABF(abf, return_obj=True)
    stim_x, stim_y, stim_c, obj = loadABF(stim_abf, return_obj=True)
    stim_scale_factor = obj2._dacSection.fDACFileScale[0]
    return run_analysis(f_x, f_y, f_c, stim_x, stim_y, stim_c, stim_scale_factor=stim_scale_factor, line_baseline=line_baseline, compute_inst_freq=compute_inst_freq, mask_spikes=mask_spikes, plot=plot, mask_sweeps=mask_sweeps)


def run_analysis(f_x, f_y, f_c, stim_x, stim_y, stim_c, stim_scale_factor=1, line_baseline=True, compute_inst_freq=True, mask_spikes=True, plot=False, mask_sweeps=False):
    """Run EPSP analysis over arrays of data. In this case, since we are doing unpaired analysis, the stim and response are passed in as separate arrays.

    Args:
        f_x (np.array): time axis of the voltage response
        f_y (np.array): voltage response data
        f_c (np.array): current command data
        stim_x (np.array): time axis of the stimulus
        stim_y (np.array): stimulus data
        stim_c (np.array): stimulus current data
        line_baseline (bool, optional): Applies linear regression to baseline the sweep,
                                        accounting for voltage drift. Defaults to True.
        compute_inst_freq (bool, optional): Compute the inst. freq. response for each sweep. 
                                            Takes a long time and can be disabled. Defaults to True.
        mask_spikes (bool, optional): Masks the spikes in the response. Defaults to True.
        plot (bool, optional): plot the gathered data. Defaults to False.
        mask_sweeps (bool, optional): Masks the sweeps that have a variance ratio greater than 1.5. Defaults to False.

    Returns:
        dict: containing the resp means and FFT data
    

    """
    
    #drop stim_y where there is no f_y sweeps
    stim_y = stim_y[:f_y.shape[0]]
    stim_x = stim_x[:f_y.shape[0]]

    #apply scaling
    sr = int(1/(f_x[0,1]-f_x[0, 0]))
    stim_y = stim_y * stim_scale_factor 
    #Baseline the data
    t0 = int(sr * 2)
    t1 = int(sr * 3.5)
    #compute the variance along the sweep axis
    pre_stim_var = np.var(f_y[:,:t0], axis=1)
    #compute the variance along the sweep axis
    post_stim_var = np.var(f_y[:,t0:t1], axis=1)
    #compute the ratio of the two
    var_ratio = pre_stim_var/post_stim_var


    #get rmp
    rmp = rmp_mode(f_y[:, :], stim_y)

    #mask the sweeps that have a variance ratio greater than 1.5
    mask = var_ratio > 1.5
    #apply the mask to the data
    if mask_sweeps:
        f_y[mask, :] = 0

    #find spikes
    spike_features = {}
    spike_time = {}
    #using ipfx
    sp = SpikeFeatureExtractor(dv_cutoff=20, min_peak=-5, filter=0) #using default values, min peak set high to make sure it does not over detect
    for i in range(f_y.shape[0]):
        #extract the spikes
        features = sp.process(f_x[i], f_y[i], stim_y[i])
        #if there are spikes
        if features.empty==False:
            #get the spike_count
            spike_count = len(features['peak_t'])
            spike_features.update({f'spike_count_sweep_{compute_real_sweep_num(i+1)}':spike_count})
            #add in the slope/upstroke/downstroke
            spike_features.update({f'upstroke_sweep_{compute_real_sweep_num(i+1)}':features['upstroke'].values[0]})
            spike_features.update({f'downstroke_sweep_{compute_real_sweep_num(i+1)}':features['downstroke'].values[0]})
            
            #add in the spike time
            spike_time.update({f'sweep_{compute_real_sweep_num(i+1)}_spike_time':features['peak_t'].values[0]})
            if mask_spikes:
                #get the idx of the threshold
                spike_times = np.array([np.argmin(np.abs(f_x[i] - x)) for x in features['threshold_t']])
                #get the trough idx
                trough_times = np.array([np.argmin(np.abs(f_x[i] - x)) for x in features['fast_trough_t']])
                #mask between the threshold and the trough using the point of the threshold
                for j in range(len(spike_times)):
                    #mask the spike, take 0.5ms before the threshold and 1ms after the trough
                    f_y[i][spike_times[j]-int(0.0005*sr):trough_times[j]+int(0.001*sr)] = f_y[i, spike_times[j]-int(0.0005*sr)]
        else:
            spike_features.update({f'spike_count_sweep_{compute_real_sweep_num(i+1)}':0})
            spike_features.update({f'upstroke_sweep_{compute_real_sweep_num(i+1)}':np.nan})
            spike_features.update({f'downstroke_sweep_{compute_real_sweep_num(i+1)}':np.nan}) 
            spike_time.update({f'sweep_{compute_real_sweep_num(i+1)}_spike_time':np.nan})


    #baseline using the mean
    fyb = f_y - np.nanmean(f_y[:, :t0], axis=1, keepdims=True)
    if line_baseline: # if the user wants to baseline the data with a linear regression
        fy_lin = [] #create an empty list to store the baseline corrected data
        for step in np.arange(f_x.shape[0]): #loop through the sweeps
            temp_fy_lin = stats.linregress(np.hstack((f_x[0,:t0], f_x[0,-t0:])), np.hstack((fyb[step,:t0], fyb[step,-t0:])))
            fy_lin.append((temp_fy_lin.slope * f_x[step] + temp_fy_lin.intercept)) #append the baseline corrected data to the list
        fy_lin = np.vstack(fy_lin) #convert the list to a numpy array
        fyb = fyb - fy_lin #subtract the linear baseline from the data


    #compute the freq-dependent impedence response
    fft_x,fft_resp = compute_impedence_freq(f_x, fyb, stim_y, [0, 20], f_step=100)


    #now we are going to step through the sweeps and compute some metrics
    # gonna create a million lists to store the data, cause im lazy
    f_stim = [] #list to store the stimulus frequency
    pointwise_diff = [] #list to store the pointwise difference between the stimulus and the response, UNUSED, only for paired data
    stim_amp = [] #//UNUSED FOR NOW// list to store the stimulus amplitude
    time_ind = np.arange(stim_x.shape[1]) #create an array of time indices
    binned_t = np.split(time_ind, 10) #split the time indices into 10 bins
    sweepwise_hz = [] #The overall stim freq per sweep
    f_sweepwise_resp = [] #The overall response per sweep
    f_resp = [] #The overall inst. freq. response
    sweep_wise_offset = [] #The overall offset per sweep
    sweep_wise_offset_params = [] #The overall offset params per sweep
    sweep_wise_peak_mean = [] #The overall peak mean per sweep
    sweep_wise_peaks = [] #The overall peaks per sweep
    sweep_wise_auc = [] #The overall area under the curve per sweep
    for sweep in np.arange(f_x.shape[0]):
        #compute the stimulus frequency
        peaks = signal.find_peaks(stim_y[sweep],width=int(sr*0.001), height=3) #find the peaks in the stimulus

        #compute the mean response
        sweepwise_hz.append(len(peaks[0])/1) #compute the frequency of the stimulus
        f_sweepwise_resp.append(np.nanmean(fyb[sweep, t0:t1][(fyb[sweep, t0:t1]>=0)])) #compute the mean of the response

        #compute the AUC of the postive response
        sweep_wise_auc.append(np.trapz(fyb[sweep, t0:t1][(fyb[sweep, t0:t1]>=0)], f_x[sweep, t0:t1][(fyb[sweep, t0:t1]>=0)]))

        #now compute the peak response
        peak_response = []
        for peak in peaks[0]: #for each peak
            #we want to take the max of the 10ms before and after the peak
            peak_response.append(np.nanmax(fyb[sweep, peak-int(sr*0.01):peak+int(sr*0.01)]))
        sweep_wise_peak_mean.append(np.mean(peak_response)) #compute the mean peak response
        sweep_wise_peaks.append(np.copy(peak_response)) #store the peaks


        #compute the inst. freq. of the sweep by using elephant
        if compute_inst_freq: #if the user wants to compute the inst. freq.
            spike_times = stim_x[sweep,peaks[0]] #Pull the spike times
            spike_train = neo.SpikeTrain(spike_times, t_stop=13*pq.s, units=pq.s) #turn the epsp incoming into a spike train, t_stop is meaning less
            kernel = kernels.GaussianKernel(sigma=300 * pq.ms) #create a gaussian kernel
            i_r = np.hstack(statistics.instantaneous_rate(spike_train, (1/sr)*pq.s, kernel=kernel)[:,0].tolist()) #compute the inst. freq. of the sweep
            for bin in binned_t:
                idx_to_use = bin
                f_resp.append(np.nanmean(fyb[sweep, idx_to_use][fyb[sweep, idx_to_use]>0]))
                f_stim.append(np.nanmean(i_r[idx_to_use]))
        else:
            #if the user disables the inst. freq. we need to fill the means with zeros so the data is the same shape
            f_stim.append(0)
            f_resp.append(0)
            

        #Now fitting the exp decay
        #find the last point of nonzero in stim y
        try:
            last_point = np.where(stim_y[sweep]>0)[0][-1]
            #take 500ms after the last point
            sweep_wise_offset.append(fyb[sweep][last_point:last_point+int(sr*1)])
            #curve fit to the last 1second
            out = curve_fit(exp_decay, stim_x[0][last_point:last_point+int(sr*1)] - stim_x[0][last_point], fyb[sweep][last_point:last_point+int(sr*1)],
            bounds=([0, 1/0.5, -np.inf], [20, 1/0.001, np.inf]), p0=[3, 10, 0], maxfev=10000, xtol=None, ftol=1e-12)
            sweep_wise_offset_params.append(out[0])
        except:
            sweep_wise_offset.append(np.nan)
            sweep_wise_offset_params.append([np.nan, np.nan, np.nan])
        
    
    
    #compute means
    f_resp = np.array(f_resp)
    f_resp_mean = []
    bins_in = np.arange(0, 30, 2.5)
    hist, bins = np.histogram(f_stim, bins_in)
    indices = np.digitize(f_stim, bins, right=True)
    uni = np.unique(indices)
    bins_in = np.hstack((0, bins_in, 30))
    x_data =  bins_in[uni]
    sweep_wise_offset_params =  np.array(sweep_wise_offset_params)
    for val in np.unique(indices):
        idx_to_use = np.ravel(np.argwhere(indices==val))
        f_resp_mean.append(np.nanmean(f_resp[idx_to_use]))

    #compute the QC metrics
    qc_metrics = run_qc(fyb, np.round(stim_y))
    qc_metrics_dict = {'mean_rms':qc_metrics[0], 'max_rms':qc_metrics[1], 'mean_drift':qc_metrics[2], 'max_drift':qc_metrics[3]}
    
    #compile some metadata
    meta_data = { "filename": '', "file_path": '', 
                "protocol": '', "protocol_scale_factor": stim_scale_factor, 'sweeps_measured': f_y.shape[0],
                "stim_filename": '', "stim_file_path": '',
                "abf_name":'', "stim_abf":'', 
                "mean_rms":qc_metrics[0], "max_rms":qc_metrics[1], 
                "mean_drift":qc_metrics[2], "max_drift":qc_metrics[3], "rmp":rmp,
                }
    meta_data.update({f"sweep_{compute_real_sweep_num(i+1)}_variance_ratio":var_ratio[i] for i in range(len(var_ratio))}) #add the variance ratio to the metadata. account for sweep number
    meta_data.update({f"sweep_masked_{compute_real_sweep_num(i+1)}":mask[i] for i in range(len(mask))}) #add the mask to the metadata. account for sweep number

    if plot:
        plt.clf()
        plt.figure()
        plt.scatter(f_stim, f_resp, alpha=0.1)
        plt.plot(x_data, f_resp_mean)
        plt.ylabel("voltage resp amplitude (mV)")
        plt.xlabel("input freq")
        plt.legend()
        plt.pause(10)

    #convert the metric lists to numpy arrays
    sweepwise_hz = np.array(sweepwise_hz)
    sweep_wise_peak_mean = np.array(sweep_wise_peak_mean)
    sweep_wise_auc = np.array(sweep_wise_auc)
    f_resp_mean = np.array(f_resp_mean)
    f_resp = np.array(f_resp)
    f_stim = np.array(f_stim)
    f_sweepwise_resp = np.array(f_sweepwise_resp)

    #for the peak_response we need to pad the arrays so they are the same size
    sweep_wise_peaks = equal_array_size_from_list(sweep_wise_peaks)
    #turn it into a dict, each row is a sweep, each column is a datapoint
    sweep_wise_peaks_raw ={}
    for i in range(sweep_wise_peaks.shape[0]):
        for j in range(sweep_wise_peaks.shape[1]):
            sweep_wise_peaks_raw.update({f"sweep_{compute_real_sweep_num(i+1)}_peak_{j+1}":sweep_wise_peaks[i,j]})
    #also compute the mean across the sweeps, for each peak
    peak_wise_means = np.nanmean(sweep_wise_peaks, axis=0)
    #turn it into a dict, each row is a sweep, each column is a datapoint
    peak_wise_means_raw ={f"peak_{j+1}_mean":peak_wise_means[j] for j in range(len(peak_wise_means))}
    #MASK out the metrics that have failed sweeps
    if mask_sweeps:
        f_sweepwise_resp[mask] = np.nan
        sweep_wise_auc[mask] = np.nan
        sweep_wise_peak_mean[mask] = np.nan
    #f_resp_mean[mask] = np.nan

    #for the offset params get the mean values into a dict
    sweep_wise_offset_params_mean = {f"a_mean":np.nanmean(sweep_wise_offset_params[:,0]), "tau_mean": 1/np.nanmean(sweep_wise_offset_params[:,1]), "c_mean":np.nanmean(sweep_wise_offset_params[:,2])}
    #also make a dict for each sweep
    sweep_wise_offset_params_dict = {}
    for j, i in enumerate(sweep_wise_offset_params):
        sweep_wise_offset_params_dict.update({f"a_sweep_{compute_real_sweep_num(j+1)}":i[0], f"tau_sweep_{compute_real_sweep_num(j+1)}":1/i[1], f"c_sweep_{compute_real_sweep_num(j+1)}":i[2]}) 

    #return the results as a dict for easy usage
    #return the results as a dict for easy usage
    dict_return = {"f_stim":f_stim, #the inst. freq. of the stimulus
    "f_resp":f_resp, #the inst. freq. of the response
    "f_resp_mean":f_resp_mean, #the mean inst. freq. of the response
    "x_data":x_data,  #the x data for the mean
    "sweep_wise_offset":sweep_wise_offset,  #the offset (exp decay fit) of the response
    "sweep_wise_offset_params_mean": sweep_wise_offset_params_mean,  #the offset (exp decay fit) params of the response
    "sweep_wise_offset_params_dict": sweep_wise_offset_params_dict,  #the offset (exp decay fit) params of the response for each sweep
    "sweepwise_hz":sweepwise_hz, #the overall stim freq per sweep
    "f_sweepwise_resp":f_sweepwise_resp,  #the overall response per sweep
    "fft_x":fft_x, #the fft x data
    "fft_resp":fft_resp,#the fft response
    "sweepwise_peak_mean": sweep_wise_peak_mean, #mean (+/- 10 points) the absolute peak response per sweep
    "sweepwise_peaks_raw": sweep_wise_peaks_raw, #the raw peak responses per sweep
    "peak_wise_means": peak_wise_means_raw, #the mean of each peak across sweeps e.g we mean across all sweeps the first peak, second peak, etc.
    'sweepwise_auc':sweep_wise_auc, #the area under the curve of the response per sweep
    'qc_metrics':qc_metrics_dict, #the qc metrics of the recording
    'meta_data':meta_data, #the metadata of the recording
    'spike_features':spike_features, #the spike features
    'spike_time':spike_time}  #the spike times
    return dict_return

def main(abf_path= "I:\\_MARM AND MOUSE\\repeating_EPSP_NOISE"):
    #CHANGE THIS TO THE PATH OF THE DATA
    
    #glob seeks all files in the directory that are abf files
    glob_path = abf_path + "\\**\\*.abf"
    files = glob.glob(glob_path, recursive=True) 
    #points to the stim_file_path which is the file that contains the stimulus, since its a CUSTOM
    #stimulus we need to point to it and load it seperately
    STIM_FILE_PATH = "EPSP_8_Pulse_50_Hz.abf"

    #COMPUTE_INST_FREQ
    #if the user wants to compute the inst. freq. response, it takes a long time
    COMPUTE_INST_FREQ = False

    #lists for storing the data
    ids =[] #the id of the file
    _resp_means = [] #the mean of the inst. freq. response
    _resp_means_norm = [] #the mean of the inst. freq. response normalized by min and max
    _resp_means_sweep = [] #the mean of the response per sweep
    _resp_means_sweep_norm = [] #the mean of the response per sweep normalized by min and max
    _peak_means = [] #the mean of the peak response, per sweep
    _peak_wise_means = [] #the mean of the peak response, per peak
    _peak_full = [] #the full peak response
    _auc_means = [] #the mean of the auc response
    fft_resp_means = [] #the mean of the fft response
    decay_means = [] #the mean of the decay
    decay_full = [] #the full decay
    meta_data = [] # the metadata of the files
    spike_features = [] #the spike features
    spike_times = [] #the spike times
    sweepwise_HZ_out = []
    for p, row in enumerate(files[::-1]): #loop through the files
        print(f"processing {p/len(files)*100} %") #print the progress
        #STIM_FILE_PATH = determine_protocol_to_use(row) #determine the protocol to use
        #if the protocol is not found, skip the file
        if STIM_FILE_PATH is None:
            print(f"Could not find protocol for {row}")
            continue
        #try:
        out = run_analysis(row, STIM_FILE_PATH, compute_inst_freq=COMPUTE_INST_FREQ) #run the analysis
        #except:
            #print(f"Could not process {row}")
            #continue
        #returns the dict with the data

        ids.append([os.path.basename(row)]) #store the id for reference later

        #Inst. freq. response
        _resp_means.append(out["f_resp_mean"]) #store the mean of the inst. freq. response
        #Inst freq. response normalized by min and max
        norm_means = np.vstack(out["f_resp_mean"]) #normalize the mean of the inst. freq. response
        norm_means = np.hstack(((norm_means - np.min(norm_means)) / (np.max(norm_means) - np.min(norm_means)))).reshape(1,-1)
        _resp_means_norm.append(norm_means) #store the normalized mean of the inst. freq. response

        #Sweepwise mean response
        _resp_means_sweep.append(out["f_sweepwise_resp"]) #store the mean of the response per sweep
        sweep_means = np.vstack(out["f_sweepwise_resp"]) #normalize the mean of the response per sweep
        sweep_means = np.hstack(((sweep_means - np.min(sweep_means)) / (np.max(sweep_means) - np.min(sweep_means))))
        _resp_means_sweep_norm.append(sweep_means) #store the normalized mean of the response per sweep

        #Sweepwise peak response
        _peak_means.append(out["sweepwise_peak_mean"]) #store the mean of the peak response
        _peak_full.append(out["sweepwise_peaks_raw"])
        _peak_wise_means.append(out["peak_wise_means"])
        #Sweepwise auc response
        _auc_means.append(out["sweepwise_auc"]) #store the mean of the auc response

        #FFT response
        fft_resp_means.append(np.amax(out['fft_resp'], axis=0)) #store the mean of the fft response

        #Decay response
        decay_means.append(out["sweep_wise_offset_params_mean"]) #store the mean of the decay response
        decay_full.append(out["sweep_wise_offset_params_dict"]) #store the full decay response
        #store the sweepwise metadata
        meta_data.append(out['meta_data'])

        #store the spike features
        spike_features.append(out['spike_features'])

        #store the spike times
        spike_times.append(out['spike_time'])
        
        if len(out['sweepwise_hz']) > len(sweepwise_HZ_out):
            sweepwise_HZ_out = np.copy(out['sweepwise_hz']).astype(str)

    fft_x = out['fft_x'] #store the fft x data, since its the same for all files we can just take it from the last file
    #Stack the data into numpy arrays for easy usage
    #each row is a file, column is a datapoint (normally a freq)
    fft_resp_means = equal_array_size_from_list(fft_resp_means)
    _resp_means = equal_array_size_from_list(_resp_means)
    _resp_means_norm = equal_array_size_from_list(_resp_means_norm)
    _resp_means_sweep = equal_array_size_from_list(_resp_means_sweep)
    _resp_means_sweep_norm = equal_array_size_from_list(_resp_means_sweep_norm)
    _auc_means = equal_array_size_from_list(_auc_means)
    _peak_means = equal_array_size_from_list(_peak_means)
    #also fix the dicts
    _peak_full = equal_dict_from_list_of_dicts(_peak_full)
    decay_full = equal_dict_from_list_of_dicts(decay_full)
    decay_means = equal_dict_from_list_of_dicts(decay_means)
    _peak_wise_means = equal_dict_from_list_of_dicts(_peak_wise_means)
    spike_features = equal_dict_from_list_of_dicts(spike_features)
    spike_times = equal_dict_from_list_of_dicts(spike_times)


    #fft_resp_means = np.vstack(fft_resp_means)
    ids = np.hstack(ids)

    #now dump the data into a dataframe and save it as a excel file
    with pd.ExcelWriter(abf_path+f'\\{os.path.basename(STIM_FILE_PATH)}data_sheet.xlsx') as writer:
        ## META DATA
        meta_data = pd.DataFrame(data=meta_data, index=ids)
        col_order = ['filename', 'file_path', 'protocol', 'protocol_scale_factor',
       'stim_filename', 'stim_file_path', 'abf_name', 'stim_abf', 'mean_rms',
       'max_rms', 'mean_drift', 'max_drift', 'rmp']
        #then the rest alphabetically
        col_order.extend([x for x in meta_data.columns if x not in col_order])
        meta_data = meta_data.reindex(col_order, axis=1)
        
        ## SPIKE FEATURES
        spike_features = pd.DataFrame(data=spike_features, index=ids)
        #sort the columns alphabetically
        spike_features = spike_features.reindex(sorted(spike_features.columns), axis=1)                                                         
        
        ## SPIKE TIMES
        spike_times = pd.DataFrame(data=spike_times, index=ids)
        

        ## INST FREQ.
        if COMPUTE_INST_FREQ:
            unpaired_inst_freq_means = pd.DataFrame(data=_resp_means, index=ids) 
            unpaired_inst_freq_means.to_excel(writer, sheet_name='inst_freq_means')

            unpaired_inst_freq_means_normed2 = pd.DataFrame(data=_resp_means_norm, index=ids)
            unpaired_inst_freq_means_normed2.to_excel(writer, sheet_name='inst_freq_means_normed')

        ## DECAY, Time constant
        unpaired_decay_means = pd.DataFrame(data=decay_means, index=ids)
        unpaired_decay_means_full = pd.DataFrame(data=decay_full, index=ids)
        #sort the columns alphabetically
        unpaired_decay_means_full = unpaired_decay_means_full.reindex(sorted(unpaired_decay_means_full.columns), axis=1)
        

        ## SWEEPWISE MEAN
        unpaired_sweepwise_resp_means = pd.DataFrame(data=_resp_means_sweep, index=ids, columns=[f"{x}_sweepwise_mean" for x in sweepwise_HZ_out])
        unpaired_sweepwise_resp_means_normed = pd.DataFrame(data=_resp_means_sweep_norm, index=ids, columns=[f"{x}_sweepwise_mean_NORMALIZED" for x in sweepwise_HZ_out])
        

        ## SWEEPWISE PEAK
        unpaired_sweepwise_peak_means = pd.DataFrame(data=_peak_means, index=ids, columns=[f"{x}_sweepwise_peak_mean" for x in sweepwise_HZ_out])
        

        ## PEAK WISE MEAN
        unpaired_peak_wise_means = pd.DataFrame(data=_peak_wise_means, index=ids,)
        

        ## SWEEPWISE PEAK FULL
        unpaired_sweepwise_peak_full = pd.DataFrame(data=_peak_full, index=ids)

        ## SWEEPWISE AUC
        unpaired_sweepwise_auc_means = pd.DataFrame(data=_auc_means, index=ids, columns=[f"{x}_sweepwise_auc" for x in sweepwise_HZ_out])

        ## FFT
        unpaired_fft_resp_means = pd.DataFrame(data=fft_resp_means, index=ids, columns=fft_x)


        #Merge the meta_data, spike_features, spike_times, unpaired_decay_means, unpaired_peak_wise_means,  unpaired_sweepwise_peak_means and _auc_means into the same dataframe
        full_df = meta_data.merge(spike_features, left_index=True, right_index=True)
        full_df = full_df.merge(spike_times, left_index=True, right_index=True, suffixes=('', '_spike_time'))
        full_df = full_df.merge(unpaired_decay_means, left_index=True, right_index=True, suffixes=('', '_decay'))
        full_df = full_df.merge(unpaired_sweepwise_resp_means, left_index=True, right_index=True, suffixes=('', '_sweepwise_resp'))
        full_df = full_df.merge(unpaired_peak_wise_means, left_index=True, right_index=True, suffixes=('', '_peakwise_means'))
        full_df = full_df.merge(unpaired_sweepwise_peak_means, left_index=True, right_index=True, suffixes=('', '_sweepwise_peak_means'))
        full_df = full_df.merge(unpaired_sweepwise_auc_means, left_index=True, right_index=True, suffixes=('', '_auc'))
        #write the sheets
        full_df.to_excel(writer, sheet_name='full_sheet')
        meta_data.to_excel(writer, sheet_name='meta_data')
        spike_features.to_excel(writer, sheet_name='spike_features')
        spike_times.to_excel(writer, sheet_name='spike_times')
        
        unpaired_decay_means.to_excel(writer, sheet_name='time_constant_decay_means')
        unpaired_decay_means_full.to_excel(writer, sheet_name='time_constant_decay_full')
        unpaired_sweepwise_resp_means.to_excel(writer, sheet_name='sweepwise_resp_means')
        unpaired_sweepwise_resp_means_normed.to_excel(writer, sheet_name='sweepwise_resp_means_normed')
        unpaired_sweepwise_peak_means.to_excel(writer, sheet_name='sweepwise_peak_means')
        unpaired_peak_wise_means.to_excel(writer, sheet_name='peak_wise_means')
        
        unpaired_sweepwise_peak_full.to_excel(writer, sheet_name='sweepwise_peak_full')
        
        unpaired_sweepwise_auc_means.to_excel(writer, sheet_name='sweepwise_auc_means')
        
        unpaired_fft_resp_means.to_excel(writer, sheet_name='fft_resp_means')

    plt.pause(10)


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



if __name__ == "__main__":
    main_folder =  "I:\\_MARM AND MOUSE\\EPSP_8 Pulse_50 Hz"
    main(main_folder)
    #find all the subfolders
    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir() ]
    #loop through the subfolders
    #for fold in subfolders:
        #main(fold)
