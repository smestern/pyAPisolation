###########
# This script builds the database for the web visualization
# It can take a long time to run, so it is recommended to run it in the background
# Here we we are going to take a folder of ABF or NWB files, and extract some features
# we will choose to use a custom backend or ipfx to extract the features
# from each file. We will then save the features in a database file.
# The database file will be used by the web visualization to display the data.
# The database file is a JSON file, or csv
###########
# Import libraries
import os
import sys
import json
import glob
import argparse
import pandas as pd
import numpy as np
import logging
from functools import partial
import copy
import joblib
import matplotlib.pyplot as plt
import scipy.stats
from multiprocessing import pool, freeze_support
# Import ipfx
import ipfx
import ipfx.script_utils as su
from ipfx.stimulus import StimulusOntology
import allensdk.core.json_utilities as ju
from ipfx.bin import run_feature_collection
from ipfx import script_utils as su
from ipfx.sweep import SweepSet, Sweep
import ipfx.stim_features as stf
import ipfx.stimulus_protocol_analysis as spa
import ipfx.data_set_features as dsf
import ipfx.time_series_utils as tsu
import ipfx.feature_vectors as fv


# Import custom functions
from pyAPisolation import patch_utils
from pyAPisolation.utils import arg_wrap
from pyAPisolation.loadNWB import loadNWB, GLOBAL_STIM_NAMES
try:
    from pyAPisolation.dev import stim_classifier as sc
except:
    print("Could not import stim_classifier")

# ==== GLOBALS =====
_ONTOLOGY = ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE)
_UNIT_ONTOLOGY = {'amp': ['amp', 'ampere', 'amps', 'amperes', 'A'],'volt': ['volt', 'v', 'volts', 'V'], 'sec': ['sec', 's', 'second', 'seconds', 'secs', 'sec']}
log = logging.getLogger(__name__)



def glob_files(folder, ext="nwb"):

    #this function will take a folder and a file extension, and return a list of files
    # with that extension in that folder
    return glob.glob(folder + "/**/*." + ext, recursive=True)


def run_analysis(folder, backend="ipfx", outfile='out.csv', ext="nwb", parallel=False):

    files = glob_files(folder)[::-1]
    file_idx = np.arange(len(files))
     
    if backend == "ipfx":
        # Use ipfx to extract features
        #get_stimulus_protocols(files)
        GLOBAL_STIM_NAMES.stim_inc = ['']
        GLOBAL_STIM_NAMES.stim_exc = []
        get_data_partial = partial(data_for_specimen_id,
                                passed_only=False,
                                data_source='filesystem',
                                ontology=None,
                                file_list=files)
        if parallel == True:
            #Run in parallel
            parallel = joblib.cpu_count()
        results = joblib.Parallel(n_jobs=1, backend='multiprocessing')(joblib.delayed(get_data_partial)(specimen_id) for specimen_id in file_idx)
        
    elif backend == "custom":
        # Use custom backend to extract features, this uses the spike_finder and patch_utils method, on the very far backend it uses ipfx, so feature values will more or less be the same
        results = []
        for f in files:
            # Extract features from each file
            result = feature_extraction.extract_features(f)
            results.append(result)
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(outfile, index=False)

    results = pd.DataFrame().from_dict(results).set_index('specimen_id')
    return results


def main():
    #main function to be called when running this script
    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Build database for web visualization')
    parser.add_argument('folder', type=str, default=None, help='Folder containing ABF or NWB files')
    # add a data_folder argument which will be the folder where the data is stored is mut
    parser.add_argument('--backend', type=str, default="ipfx", help='Backend to use for feature extraction')
    parser.add_argument('--outfile', type=str, default="out.csv", help='Output file name') 
    parser.add_argument('--ext', type=str, default="nwb", help='File extension to search for in the folder')
    parser.add_argument('--data_folder', type=str, default=None, help='Folder containing ABF or NWB files')
    
    parser = arg_wrap(parser, cli_prompt=True)  # Wrap the parser to catch exceptions
    args = parser.parse_args()
    #args should have folder or data_folder
    if args.folder is None and args.data_folder is None:
        raise ValueError("You must provide a folder with ABF or NWB files, either with the folder argument or the --data_folder argument")
    if args.data_folder is not None: #data_folder will supercede folder
        args.folder = args.data_folder
    # Run analysis
    run_analysis(args.folder, args.backend, args.outfile)



#======== IPFX functions ===========
#clone the function from ipfx/stimulus_protocol_analysis.py
# here we will modify it to handle test pulses intelligently, then overwrite the function in ipfx for this session
def get_stim_characteristics(i, t, test_pulse=True, start_epoch=None, end_epoch=None, test_pulse_length=0.250):
    """
    Identify the start time, duration, amplitude, start index, and end index of a general stimulus.
    """
    fs = 1/(t[1]  - t[0])
    di = np.diff(i)
    di_idx = np.flatnonzero(di)   # != 0
    start_idx_idx = 0
    
    
    if len(di_idx[start_idx_idx:]) == 0:    # if no stimulus is found
        return None, None, 0.0, None, None

    #here we will check if the first up/down is a test pulse, and skip it if it is
    #we are assuming that the test pulse is within the first 250ms of the stimulus
    #TODO make this more robust
    if len(di_idx) > 3: # if there are more than 3 up/down transitions, there is probably a test pulse
        if (di_idx[1]) < test_pulse_length*fs: # skip the first up/down (test pulse) if present, and with in the first 250ms
            start_idx_idx = 2
        else:
            start_idx_idx = 0
    elif len(di_idx) < 3:
        start_idx_idx = 0

    start_idx = di_idx[start_idx_idx] + 1   # shift by one to compensate for diff()
    end_idx = di_idx[-1]
    if start_idx >= end_idx: # sweep has been cut off before stimulus end
        return None, None, 0.0, None, None

    start_time = float(t[start_idx])
    duration = float(t[end_idx] - t[start_idx-1])

    stim = i[start_idx:end_idx+1]

    peak_high = max(stim)
    peak_low = min(stim)

    if abs(peak_high) > abs(peak_low):
        amplitude = float(peak_high)
    else:
        amplitude = float(peak_low)

    return start_time, duration, amplitude, start_idx, end_idx

ipfx.stim_features.get_stim_characteristics = get_stim_characteristics

def parse_long_pulse_from_dataset(data_set):
    sweeps = []
    start_times = []
    end_times = []
    for sweep in np.arange(len(data_set.dataY)):
        i = data_set.dataC[sweep]*1
        t = data_set.dataX[sweep]
        v = data_set.dataY[sweep]
        dt = t[1] - t[0]
        #if its not current clamp
        if match_unit(data_set.sweepMetadata[sweep]['stim_dict']["unit"]) != "amp":
            continue
        #if the sweep v is in volts, convert to mV, ipfx wants mV
        if match_unit(data_set.sweepMetadata[sweep]['resp_dict']["unit"]) == "volt":
            #sometimes the voltage is in volts, sometimes in mV, this is a hack to fix that
            if np.max(v) > 500 and np.min(v) < -500:
                #possibly in nV or something else, convert to mV anyway
                v = v/1000
            elif np.max(v) < 1 and np.min(v) > -1:
                #probably in volts, convert to mV
                v = v*1000
        
        #if the sweep i is in amps, convert to pA, ipfx wants pA
        if match_unit(data_set.sweepMetadata[sweep]['stim_dict']["unit"])=="amp":
            if np.max(i) < 0.1 and np.min(i) > -0.1:
                #probably in amp, convert to picoAmps
                i = np.rint(i*1000000000000).astype(np.float32)
            else:
                #probably in pA already
                i = np.rint(i).astype(np.float32)
            #sometimes i will have a very small offset, this will remove it
            i[np.logical_and(i < 0.5, i > -0.5)] = 0
        if match_protocol(i, t) != "Long Square":
            continue
        start_time, duration, amplitude, start_idx, end_idx = get_stim_characteristics(i, t)
        if start_time is None:
            continue
        #construct a sweep obj
        start_times.append(start_time)
        end_times.append(start_time+duration)
        sweep_item = Sweep(t, v, i, clamp_mode="CurrentClamp", sampling_rate=int(1/dt), sweep_number=sweep)
        sweeps.append(sweep_item)
    return sweeps, start_times, end_times

def data_for_specimen_id(specimen_id, passed_only, data_source, ontology, file_list=None, amp_interval=20, max_above_rheo=100, debug=True):
    
    result = {}
    result["specimen_id"] = file_list[specimen_id]

    try:
        #this is a clone of the function in ipfx/bin/run_feature_collection.py,
        # here we are gonna try to use it to handle data that may not be in an NWB format IPFX can handle
        _, _, _, _, data_set = loadNWB(file_list[specimen_id], return_obj=True)
        if data_set is None or len(data_set.dataY)<1:
            return result
        #here we are going to perform long square analysis on the data, 
        #ipfx does not play nice with many NWBs on dandi, so we are going to link into the lower level functions
        #and do the analysis ourselves
        #hopefully this will be fixed in the future and we can use ipfx for this
        sweeps = []
        start_times = []
        end_times = []
        
        debug_log = {}
        for sweep in np.arange(len(data_set.dataY)):
            i = np.nan_to_num(data_set.dataC[sweep]*1)
            t = data_set.dataX[sweep]
            v = np.nan_to_num(data_set.dataY[sweep])
            dt = t[1] - t[0]
            #if its not current clamp
            if match_unit(data_set.sweepMetadata[sweep]['stim_dict']["unit"]) != "amp":
                logging.debug(f"sweep {sweep} is not current clamp")
                debug_log[sweep] = f"not current clamp, found units of {match_unit(data_set.sweepMetadata[sweep]['stim_dict']['unit'])}"
                continue
            #if the sweep v is in volts, convert to mV, ipfx wants mV
            if match_unit(data_set.sweepMetadata[sweep]['resp_dict']["unit"]) == "volt":
                #sometimes the voltage is in volts, sometimes in mV,  even thought it is logged as bolts this is a hack to fix that
                if np.max(v) > 500 and np.min(v) < -500:
                    #possibly in nV or something else, convert to mV anyway
                    v = v/1000
                elif np.max(v) < 1 and np.min(v) > -1:
                    #probably in volts, convert to mV
                    v = v*1000
            
            #if the sweep i is in amps, convert to pA, ipfx wants pA
            if match_unit(data_set.sweepMetadata[sweep]['stim_dict']["unit"])=="amp":
                if np.max(i) < 0.1 and np.min(i) > -0.1:
                    #probably in amp, convert to picoAmps
                    i = i*1000000000000
                
                #probably in pA already
                #i[np.logical_and(i < 5, i > -5)] = 0

            #try to figure out if this is a long square
            if match_protocol(i, t) != "Long Square":
                logging.debug(f"skipping sweep {sweep} because it is not a long square")
                debug_log[sweep] = "likely not a long square"
                continue

            start_time, duration, amplitude, start_idx, end_idx = get_stim_characteristics(i, t)

            if QC_voltage_data(t, v, i) == 0:
                logging.debug(f"skipping sweep {sweep} because it failed QC")
                debug_log[sweep] = "failed QC"
                continue
            #construct a sweep obj
            start_times.append(start_time)
            end_times.append(start_time+duration)
            sweep_item = Sweep(t, v, i, clamp_mode="CurrentClamp", sampling_rate=int(1/dt), sweep_number=sweep)
            sweeps.append(sweep_item)
        if debug:
            for sweep in debug_log.keys():
                print(f"sweep {sweep} failed QC because it was {debug_log[sweep]}")
                if debug_log[sweep] == "failed QC":
                    plt.plot(data_set.dataX[sweep], data_set.dataY[sweep], label=f"{sweep} {debug_log[sweep]}", c='r')
                else:
                    #plt.plot(data_set.dataX[sweep], data_set.dataY[sweep], label=f"{sweep} {debug_log[sweep]}", c='k')
                    continue
                
            #plt.legend()
            plt.pause(0.2)
        if len(sweeps) < 1:
            return result
        #get the most common start and end times
        start_time = scipy.stats.mode(np.array(start_times))[0][0]
        end_time = scipy.stats.mode(np.array(end_times))[0][0]
        #index out the sweeps that have the most common start and end times
        idx_pass = np.where((np.array(start_times) == start_time) & (np.array(end_times) == end_time))[0]
        sweeps = SweepSet(np.array(sweeps, dtype=object)[idx_pass].tolist())

        lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(
            sweeps,
            start=start_time , #if the start times are not the same, this will fail
            end=end_time, #if the end times are not the same, this will fail
            min_peak=-25,
        )
        lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx,
            subthresh_min_amp=-100.0)
        if np.mean(start_times) < 0.01:
           lsq_an.sptx.baseline_interval = np.mean(start_times)*0.1
           lsq_an.sptx.sag_baseline_interval = np.mean(start_times)*0.1
        lsq_features = lsq_an.analyze(sweeps)

        result.update({
                "input_resistance": lsq_features["input_resistance"],
                "tau": lsq_features["tau"],
                "v_baseline": lsq_features["v_baseline"],
                "sag_nearest_minus_100": lsq_features["sag"],
                "sag_measured_at": lsq_features["vm_for_sag"],
                "rheobase_i": int(lsq_features["rheobase_i"]),
                "fi_linear_fit_slope": lsq_features["fi_fit_slope"],
            })
        # Identify suprathreshold set for analysis
        sweep_table = lsq_features["spiking_sweeps"]
        mask_supra = sweep_table["stim_amp"] >= lsq_features["rheobase_i"]
        sweep_indexes = fv._consolidated_long_square_indexes(sweep_table.loc[mask_supra, :])
        amps = np.rint(sweep_table.loc[sweep_indexes, "stim_amp"].values - lsq_features["rheobase_i"])
        spike_data = np.array(lsq_features["spikes_set"])

        for amp, swp_ind in zip(amps, sweep_indexes):
            if (amp % amp_interval != 0) or (amp > max_above_rheo) or (amp < 0):
                continue
            amp_label = int(amp / amp_interval)

            first_spike_lsq_sweep_features = run_feature_collection.first_spike_lsq(spike_data[swp_ind])
            result.update({"ap_1_{:s}_{:d}_long_square".format(f, amp_label): v
                                for f, v in first_spike_lsq_sweep_features.items()})

            mean_spike_lsq_sweep_features = run_feature_collection.mean_spike_lsq(spike_data[swp_ind])
            result.update({"ap_mean_{:s}_{:d}_long_square".format(f, amp_label): v
                                for f, v in mean_spike_lsq_sweep_features.items()})

            sweep_feature_list = [
                "first_isi",
                "avg_rate",
                "isi_cv",
                "latency",
                "median_isi",
                "adapt",
            ]

            result.update({"{:s}_{:d}_long_square".format(f, amp_label): sweep_table.at[swp_ind, f]
                                for f in sweep_feature_list})
            result["stimulus_amplitude_{:d}_long_square".format(amp_label)] = int(amp + lsq_features["rheobase_i"])

        rates = sweep_table.loc[sweep_indexes, "avg_rate"].values
        result.update(run_feature_collection.fi_curve_fit(amps, rates))

        #we should record the name of the stimuli used and the sweeps used, for either plotting or debugging
        result["stimulus_name"] = data_set.sweepMetadata[0]['stim_dict']['description']
        result["sweeps_used"] = sweep_indexes

        

        
    except Exception as e:
        print("error with specimen_id: ", specimen_id)
        print(e)
        plt.close()
        return result
    plt.close()
    return result

def find_time_index(t, t_0):
    """ Find the index value of a given time (t_0) in a time series (t).


    Parameters
    ----------
    t   : time array
    t_0 : time point to find an index

    Returns
    -------
    idx: index of t closest to t_0
    """
    if t[0] <= t_0 <= t[-1]: "Given time ({:f}) is outside of time range ({:f}, {:f})".format(t_0, t[0], t[-1])
    if t_0 < t[0]:
        t_0 = t[0]
    if t_0 > t[-1]:
        t_0 = t[-1]
    
    idx = np.argmin(abs(t - t_0))
    return idx

tsu.find_time_index = find_time_index

#stimulus protocol analysis functions, here we will guess what stimulus protocol was used, and affix that to the stimulus ontology later
def get_stimulus_protocols(files, ext="nwb", method='random'):
    #this function is going to take a folder and a file extension, and return a list of stimulus protocols, then guess what type of stimulus protocol was used
    #method can be random, first, or all
    #random will choose 10 random files and try to guess the stimulus protocol
    if method == 'random':
        files = np.random.choice(files, min(100, len(files)))
    elif method == 'first':
        files = files[0]
    elif method == 'all':
        pass
    #here we are gonna set the GLOBAL_STIM_NAMES filter to blank, so that we can get all the stimulus names
    GLOBAL_STIM_NAMES.stim_inc = ['']
    GLOBAL_STIM_NAMES.stim_exc = []
    classifier = sc.stimClassifier()
    stim_to_use = []
    for i, f in enumerate(files):
        _, _, _, _, data_set = loadNWB(f, return_obj=True)
        #
        #[plt.plot(x) for x in data_set.dataY]
        #plt.show()
        for j in np.arange(len(data_set.dataY)):
            sweep_meta = data_set.sweepMetadata[j]
            i = data_set.dataC[j]
            t = data_set.dataX[j]
            #stim_protocol = match_protocol(i, t) #stimulus protocol is the matching protocol 
            stim_protocol = classifier.predict(i)
            #reference mapped to the allen protocol names
            if stim_protocol is not None:
                #add stim_protocol to ontology
                stim_name_1 = sweep_meta['description']
                stim_name_2 = sweep_meta['stimulus_description']
                for stim_name in [stim_name_1, stim_name_2]:
                    if stim_name not in GLOBAL_STIM_NAMES.stim_inc:
                        if stim_name != '' and stim_name != 'N//A' and stim_name != 'NA' and stim_name != 'N/A':
                            stim_to_use.append(stim_name)
                
    GLOBAL_STIM_NAMES.stim_inc = stim_to_use   
    return copy.deepcopy(GLOBAL_STIM_NAMES)

def match_protocol(i, t, test_pulse=True, start_epoch=None, end_epoch=None, test_pulse_length=0.1):
    #this function will take a stimulus and return the stimulus protocol at least it will try
    #first we will try to match the stimulus protocol to a known protocol
    #classifier = sc.stimClassifier()
    start_time, duration, amplitude, start_idx, end_idx = get_stim_characteristics(i, t, test_pulse=test_pulse, start_epoch=start_epoch, end_epoch=end_epoch, test_pulse_length=test_pulse_length)
    #pred = classifier.decode(classifier.predict(i.reshape(1, -1)))[0]
    #if pred=="long_square":
     #   return "Long Square"
    if start_time is None:
        #if we can't find the start time, then we can't identify the stimulus protocol
        return None
    if duration > 0.25:
        #if the stimulus is longer than 500ms, then it is probably a long square
        return match_long_square_protocol(i, t, start_idx, end_idx)
    elif duration < 0.1:
        #if the stimulus is less than 100ms, then it is probably a short square
        return match_short_square_protocol(i, t)
    else:
        #check if ramp
        return match_ramp_protocol(i, t)
    

def match_long_square_protocol(i, t, start_idx, end_idx):
    #here we will do some analysis to determine if the stimulus is a long square, and if so, what the parameters are
    fs = 1/(t[1]  - t[0])

    di = np.diff(i)
    di_idx = np.flatnonzero(di)   # != 0

    if len(di_idx) == 0:
        #if there are no up/down transitions, then this is not a long square
        return None
    if len(di_idx) == 1:
        #if there is only one up/down transition, then this is not a long square
        return None
    #if len(di_idx) > 6:
        #if there are more than 6 up/down transitions, then this is (probably) not a long square
      #  return 
    #check if its a long square by fitting a line to the dataset,
    #and checking if the slope is 0
    #if the slope is 0, then it is a long square
    #if the slope is not 0, then it is not a long square
    if len(di_idx) > 6:
        y_data = i[start_idx: end_idx]
        x_data = t[start_idx: end_idx]
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x_data, y_data)
        if slope < 0.1 and p_value < 0.05 and r_value > 0.6:
            return 'Long Square'
        elif slope > 0.1 and p_value > 0.05:
            return 'Long Square'
        else:
            return None
    #ensure that the stim starts at 0, and ends at 0
    if i[0] != 0:
        return None
    if i[-1] != 0:
        return None
    

    return "Long Square"

def match_short_square_protocol(stimulus_protocol, ontology):
    #TODO: implement this function
    pass

def match_ramp_protocol(stimulus_protocol, ontology):
    #TODO: implement this function
    pass

def match_unit(unit, ontology=_UNIT_ONTOLOGY):
    #unit should be a string, if its bytes or something else, convert it to a string
    if isinstance(unit, bytes) or isinstance(unit, np.bytes_):
        unit = unit.decode('utf-8')
    elif isinstance(unit, str) != True:
        unit = str(unit)
    #this function will take a unit and return the unit ontology
    for unit_name in ontology:
        check = [unit.upper() in x.upper() for x in ontology[unit_name]]
        if np.any(check):
            return unit_name
    return None


def QC_voltage_data(t,v,i, zero_threshold=0.2, noise_threshold=10):
    #this function will take a voltage trace and return a QC score
    #Sometimes the voltage trace is not a voltage trace, but rather a current trace
    #or with IGOR / MIES generated NWB files, the sweep was aborted halfway through, and there is a large jump in the voltage trace, and a bunhc of zeros
    #this function will check for these things and return a QC score
    #if the QC score is 0, then the sweep is bad
    #if the QC score is 1, then the sweep is good
    if v is None:
        return 0
    if i is None:
        return 0
    if len(v) == 0:
        return 0
    if np.any(v > 500) or np.any(v < -500): #membrane voltages are very very unlikely to be this large, this threshold could be lowered
        return 0
    

    #check for extended periods of 0
    if np.sum(v == 0) > zero_threshold*len(v): #if more than 10% of the trace is 0, then it was probably aborted
        #this is only a problem if the current is not 0
        #check if while the voltage is 0, the current is 0
        idx_zero = np.flatnonzero(np.isclose(v, 0))
        if np.sum(i[idx_zero] != 0) > (zero_threshold/2)*len(idx_zero):
            return 0
        else:
            return 1 
        

    #check for large jumps in the voltage trace
    #dv = np.diff(v)
    #if np.any(np.abs(dv) > 1e9):
        #return 0

    #todo, more qc checks
    return 1


def build_dataset_traces(folder, ext="nwb", parallel=True):
    #this function will take a run_analysis function and return a dataset of traces
    files = glob_files(folder)[::-1]
    file_idx = np.arange(len(files))
    if parallel == True:
        #Run in parallel
        parallel = joblib.cpu_count()
    results = joblib.Parallel(n_jobs=parallel, backend='multiprocessing')(joblib.delayed(plot_data)(specimen_id, files) for specimen_id in file_idx)

def plot_data(specimen_id, file_list=None, target_amps=[-100, -20, 20, 100], debug=True):
    result = {}
    result["specimen_id"] = file_list[specimen_id]
    _, _, _, _, data_set = loadNWB(file_list[specimen_id], return_obj=True)
    if data_set is None or len(data_set.dataY)<1:
        return result
    #here we are going to perform long square analysis on the data,
            #hopefully this will be fixed in the future and we can use ipfx for this
    sweeps = []
    start_times = []
    end_times = []
    sweep_amp = []
    debug_log = {}
    for sweep in np.arange(len(data_set.dataY)):
        i = np.nan_to_num(data_set.dataC[sweep]*1)
        t = data_set.dataX[sweep]
        v = np.nan_to_num(data_set.dataY[sweep])
        dt = t[1] - t[0]
        #if its not current clamp
        if match_unit(data_set.sweepMetadata[sweep]['stim_dict']["unit"]) != "amp":
            logging.debug(f"sweep {sweep} is not current clamp")
            #debug_log[sweep] = "not current clamp"
            continue
        #if the sweep v is in volts, convert to mV, ipfx wants mV
        if match_unit(data_set.sweepMetadata[sweep]['resp_dict']["unit"]) == "volt":
            #sometimes the voltage is in volts, sometimes in mV,  even thought it is logged as bolts this is a hack to fix that
            if np.max(v) > 500 and np.min(v) < -500:
                #possibly in nV or something else, convert to mV anyway
                v = v/1000
            elif np.max(v) < 1 and np.min(v) > -1:
                #probably in volts, convert to mV
                v = v*1000
        
        #if the sweep i is in amps, convert to pA, ipfx wants pA
        if match_unit(data_set.sweepMetadata[sweep]['stim_dict']["unit"])=="amp":
            if np.max(i) < 0.1 and np.min(i) > -0.1:
                #probably in amp, convert to picoAmps
                i = i*1000000000000
            
            #probably in pA already
            #i[np.logical_and(i < 5, i > -5)] = 0

        #try to figure out if this is a long square
        if match_protocol(i, t) != "Long Square":
            logging.debug(f"skipping sweep {sweep} because it is not a long square")
            debug_log[sweep] = "likely not a long square"
            continue

        start_time, duration, amplitude, start_idx, end_idx = get_stim_characteristics(i, t)

        if QC_voltage_data(t, v, i) == 0:
            logging.debug(f"skipping sweep {sweep} because it failed QC")
            debug_log[sweep] = "failed QC"
            continue
        #construct a sweep obj
        start_times.append(start_time)
        end_times.append(start_time+duration)
        sweep_amp.append(amplitude)
        sweeps.append(Sweep(t, v, i, clamp_mode="CurrentClamp", sampling_rate=int(1/dt), sweep_number=sweep))
    
    #get the most common start and end times
    start_time = scipy.stats.mode(np.array(start_times))[0][0]
    end_time = scipy.stats.mode(np.array(end_times))[0][0]
    #index out the sweeps that have the most common start and end times
    idx_pass = np.where((np.array(start_times) == start_time) & (np.array(end_times) == end_time))[0]

    #the sweeps closest to the target amp
    idx_targets = [np.argmin(np.abs(np.array(sweep_amp) - x)) for x in target_amps]

    idx_pass = np.intersect1d(idx_pass, idx_targets)

    sweeps = np.array(sweeps, dtype=object)[idx_pass]

    sweeps = SweepSet(sweeps.tolist())
    #plot the sweeps
    fig, ax = plt.subplots(1, 1)
    for sweep in sweeps.sweeps:
        idx_start = find_time_index(sweep.t, np.clip(start_time*0.8, 0.0, np.inf))
        idx_end = find_time_index(sweep.t, np.clip(end_time*1.4, end_time, sweep.t[-1]))
        ax.plot(sweep.t[idx_start:idx_end], sweep.v[idx_start:idx_end])
    #save the figure as a svg
    plt.savefig(f"{file_list[specimen_id]}.svg")
    plt.close('all')
    print(f"saved {file_list[specimen_id]}.svg", end="\r")
    

if __name__ == "__main__":
    freeze_support()
    #call main 
    main()

    

