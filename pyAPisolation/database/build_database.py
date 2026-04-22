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
import traceback
from collections import Counter
from functools import partial
import copy
#import joblib
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
from pyAPisolation.utils import arg_wrap, debug_wrap
from pyAPisolation.loadFile.loadNWB import loadNWB, GLOBAL_STIM_NAMES
from pyAPisolation.featureExtractor import analyze_spike_times
try:
    from pyAPisolation.dev import stim_classifier as sc
except:
    print("Could not import stim_classifier")

# ==== GLOBALS =====
_ONTOLOGY = ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE)
_UNIT_ONTOLOGY = {'amp': ['amp', 'ampere', 'amps', 'amperes', 'A'],
                  'volt': ['volt', 'v', 'volts', 'V'], 
                  'sec': ['sec', 's', 'second', 'seconds', 'secs', 'sec']}
log = logging.getLogger(__name__)
logger = logging.getLogger(__name__)


# ==== ERROR CLASSIFICATION ====
# Maps substrings found in exception messages / traceback to a short error_class tag.
# These help aggregate common ipfx failure modes across a batch.
# Needles are matched in order; first match wins. Short/ambiguous substrings
# are avoided to prevent false positives from path fragments (e.g. "amp" in
# "fapl", "sag" in "message").
_ERROR_CLASS_PATTERNS = [
    ("file_not_found", ["no such file", "errno = 2", "filenotfounderror"]),
    ("no_rheobase", ["rheobase", "no spiking sweep"]),
    ("no_spikes_detected", ["no spikes detected", "empty spike", "spikes_set"]),
    ("insufficient_sweeps", ["not enough sweeps", "insufficient sweeps", "at least 2 sweeps"]),
    ("sag_calc_failed", ["sag fraction", "subthresh_min_amp", "hyperpolarizing"]),
    ("fi_fit_failed", ["fi_fit", "fi curve", "curve_fit"]),
    ("unit_mismatch", ["unit mismatch", "stim unit", "clamp_mode"]),
    ("stim_protocol", ["stim_characteristics", "long square", "stimulus protocol"]),
    ("sweep_alignment", ["start_time mismatch", "end_time mismatch"]),
    ("index_error", ["indexerror", "out of bounds", "out of range"]),
    ("nan_or_inf", ["contains nan", "contains infinity", "zero-size array"]),
]


def _classify_error(exc, tb_str):
    """Return a short error-class tag based on exception type/message/traceback."""
    msg = (str(exc) or "").lower()
    tb_lower = (tb_str or "").lower()
    # Prefer exception-message matches over traceback matches to reduce false
    # positives from substrings that happen to appear in file paths.
    for haystack in (msg, tb_lower):
        for tag, needles in _ERROR_CLASS_PATTERNS:
            for needle in needles:
                if needle in haystack:
                    return tag
    # Fallback: exception class name.
    exc_name = type(exc).__name__.lower()
    if "notfound" in exc_name or "filenotfound" in exc_name:
        return "file_not_found"
    if "index" in exc_name:
        return "index_error"
    if "value" in exc_name:
        return "value_error"
    return "unclassified"


def _deepest_frame(tb_str):
    """Return 'module.py:lineno in func' for the deepest frame in a traceback string."""
    # traceback.format_exc() frames look like: '  File "/path/x.py", line 123, in func_name'
    frames = [ln for ln in (tb_str or "").splitlines() if ln.strip().startswith("File \"")]
    if not frames:
        return ""
    last = frames[-1].strip()
    # Turn absolute path into just filename for brevity.
    try:
        # 'File "/a/b/c.py", line 12, in foo'
        path_part = last.split('"')[1]
        fname = os.path.basename(path_part)
        rest = last.split('"')[-1]  # ', line 12, in foo'
        return (fname + rest).strip(", ")
    except Exception:
        return last


def _record_exception(result, stage, exc, specimen_label=None):
    """Populate result dict with structured error fields + log via logger.exception."""
    tb_str = traceback.format_exc()
    result["error_stage"] = stage
    result["error_type"] = type(exc).__name__
    result["error_message"] = str(exc)[:500]
    result["error_origin"] = _deepest_frame(tb_str)
    # Keep only the last ~8 lines of the traceback to stay CSV-friendly.
    tb_lines = tb_str.strip().splitlines()
    result["error_traceback"] = "\n".join(tb_lines[-12:])
    result["error_class"] = _classify_error(exc, tb_str)
    logger.exception(
        "specimen=%s stage=%s class=%s: %s",
        specimen_label if specimen_label is not None else result.get("specimen_id", "?"),
        stage,
        result["error_class"],
        exc,
    )



def glob_files(folder, ext="nwb"):

    #this function will take a folder and a file extension, and return a list of files
    # with that extension in that folder
    return glob.glob(folder + "/**/*." + ext, recursive=True)


def run_analysis(folder, backend="ipfx", outfile='out.csv', ext="nwb", parallel=False):
    files = glob_files(folder)[::-1]
    file_idx = np.arange(len(files))
    import joblib
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

    # ----- summarize per-cell errors & write errors csv -----
    try:
        _log_and_write_error_summary(results, outfile)
    except Exception:
        logger.exception("failed to write error summary")

    results = pd.DataFrame().from_dict(results).set_index('specimen_id')
    return results


def _log_and_write_error_summary(results, outfile):
    """Log a per-stage/per-class breakdown of failures and write `<outfile>.errors.csv`."""
    if not results:
        return
    total = len(results)
    error_rows = []
    stage_counts = Counter()
    class_counts = Counter()
    for r in results:
        if not isinstance(r, dict):
            continue
        stage = r.get("error_stage")
        if stage:
            stage_counts[stage] += 1
            class_counts[r.get("error_class", "unclassified")] += 1
            error_rows.append({
                "specimen_id": r.get("specimen_id"),
                "error_stage": stage,
                "error_class": r.get("error_class", ""),
                "error_type": r.get("error_type", ""),
                "error_message": r.get("error_message", ""),
                "error_origin": r.get("error_origin", ""),
                "sweep_skip_summary": r.get("sweep_skip_summary", ""),
                "error_traceback": r.get("error_traceback", ""),
            })
    failed = len(error_rows)
    logger.info(
        "run_analysis: total=%d succeeded=%d failed=%d stages=%s classes=%s",
        total, total - failed, failed,
        dict(stage_counts), dict(class_counts),
    )
    if error_rows and outfile:
        err_path = str(outfile) + ".errors.csv"
        try:
            pd.DataFrame(error_rows).to_csv(err_path, index=False)
            logger.info("run_analysis: wrote per-cell error summary to %s", err_path)
        except Exception:
            logger.exception("failed writing %s", err_path)


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
    specimen_label = file_list[specimen_id]
    skip_reasons = Counter()
    # Track a small number of representative rejection details so the log can
    # show actual numbers (duration, transitions, first/last stim values, etc.)
    # rather than just a count — see diagnose_non_long_square.
    sample_rejects = {}
    SAMPLE_LIMIT_PER_TAG = 10
    # Track a small number of representative rejection details so the log can
    # show actual numbers (duration, transitions, first/last stim values, etc.)
    # rather than just a count — see diagnose_non_long_square.
    sample_rejects = {}
    SAMPLE_LIMIT_PER_TAG = 10

    # ----- stage: load -----
    try:
        _, _, _, data_set = loadNWB(file_list[specimen_id], return_obj=True)
    except Exception as e:
        _record_exception(result, "load", e, specimen_label)
        plt.close()
        return result
    if data_set is None or len(data_set.dataY) < 1:
        result["error_stage"] = "load"
        result["error_class"] = "empty_dataset"
        result["error_message"] = "loadNWB returned no sweep data"
        logger.warning("specimen=%s stage=load: empty dataset", specimen_label)
        return result

    # ----- stage: parse_sweeps -----
    sweeps = []
    start_times = []
    end_times = []
    stim_amps = []
    debug_log = {}
    try:
        for sweep in np.arange(len(data_set.dataY)):
            i = np.nan_to_num(data_set.dataC[sweep] * 1)
            t = data_set.dataX[sweep]
            v = np.nan_to_num(data_set.dataY[sweep])
            dt = t[1] - t[0]
            # if its not current clamp
            stim_unit = None
            try:
                stim_unit = match_unit(data_set.sweepMetadata[sweep]['stim_dict']["unit"])
            except Exception:
                skip_reasons["unit_parse_fail"] += 1
                debug_log[sweep] = "unit parse failed"
                continue
            if stim_unit != "amp":
                skip_reasons["not_current_clamp"] += 1
                debug_log[sweep] = f"not current clamp (unit={stim_unit})"
                continue
            # if the sweep v is in volts, convert to mV, ipfx wants mV
            if match_unit(data_set.sweepMetadata[sweep]['resp_dict']["unit"]) == "volt":
                if np.max(v) > 500 and np.min(v) < -500:
                    v = v / 1000
                elif np.max(v) < 1 and np.min(v) > -1:
                    v = v * 1000
            # if the sweep i is in amps, convert to pA, ipfx wants pA
            if stim_unit == "amp":
                if np.max(i) < 0.1 and np.min(i) > -0.1:
                    i = i * 1000000000000

            # try to figure out if this is a long square
            if match_protocol(i, t) != "Long Square":
                tag, detail = diagnose_non_long_square(i, t)
                bucket = f"not_long_square:{tag}"
                skip_reasons[bucket] += 1
                debug_log[sweep] = bucket
                samples = sample_rejects.setdefault(tag, [])
                if len(samples) < SAMPLE_LIMIT_PER_TAG:
                    samples.append({"sweep": int(sweep), **detail})
                continue

            start_time, duration, amplitude, start_idx, end_idx = get_stim_characteristics(i, t)
            if start_time is None:
                skip_reasons["no_stim_characteristics"] += 1
                debug_log[sweep] = "no stim characteristics"
                continue

            if QC_voltage_data(t, v, i) == 0:
                skip_reasons["failed_QC"] += 1
                debug_log[sweep] = "failed QC"
                continue

            i = np.round(i, decimals=1)
            start_times.append(start_time)
            end_times.append(start_time + duration)
            sweep_item = Sweep(t, v, i, clamp_mode="CurrentClamp", sampling_rate=int(1 / dt), sweep_number=sweep)
            sweeps.append(sweep_item)
            stim_amps.append(amplitude)
    except Exception as e:
        _record_exception(result, "parse_sweeps", e, specimen_label)
        result["sweep_skip_summary"] = json.dumps(dict(skip_reasons))
        plt.close()
        return result

    if debug:
        for sweep_key, reason in debug_log.items():
            logger.debug("specimen=%s sweep=%s: %s", specimen_label, sweep_key, reason)

    result["sweep_skip_summary"] = json.dumps(dict(skip_reasons))

    if len(sweeps) < 1:
        result["error_stage"] = "no_usable_sweeps"
        result["error_class"] = "no_usable_sweeps"
        top = ", ".join(f"{k}={v}" for k, v in skip_reasons.most_common(5))
        result["error_message"] = f"no sweeps survived parsing ({top})" if top else "no sweeps survived parsing"
        logger.warning("specimen=%s stage=no_usable_sweeps: %s", specimen_label, result["error_message"])
        # Emit the sample rejects so the user can see the actual stimulus stats
        # (duration, num_transitions, i_first/i_last, slope, etc.) for a few
        # representative rejected sweeps.
        for tag, samples in sample_rejects.items():
            for s in samples:
                logger.warning(
                    "specimen=%s reject_sample tag=%s %s",
                    specimen_label, tag,
                    ", ".join(f"{k}={v}" for k, v in s.items()),
                )
        plt.close()
        return result

    # ----- stage: ipfx_extractors -----
    try:
        paired_times = np.hstack((np.array(start_times).reshape(-1, 1), np.array(end_times).reshape(-1, 1)))
        paired_times = np.round(paired_times, decimals=3)
        unique_times, counts = np.unique(paired_times, axis=0, return_counts=True)
        most_common_idx = np.argmax(counts)
        start_time, end_time = unique_times[most_common_idx]
        sweeps = SweepSet(np.array(sweeps, dtype=object).tolist())

        lsq_spx, lsq_spfx = dsf.extractors_for_sweeps(
            sweeps,
            start=start_time,
            end=end_time,
            min_peak=-25,
        )
        lsq_an = spa.LongSquareAnalysis(lsq_spx, lsq_spfx, subthresh_min_amp=-200.0)
        if np.mean(start_times) < 0.01:
            lsq_an.sptx.baseline_interval = np.mean(start_times) * 0.1
            lsq_an.sptx.sag_baseline_interval = np.mean(start_times) * 0.1
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
    except Exception as e:
        _record_exception(result, "ipfx_extractors", e, specimen_label)
        plt.close()
        return result

    # ----- stage: ipfx_per_amp_features -----
    try:
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
    except Exception as e:
        _record_exception(result, "ipfx_per_amp_features", e, specimen_label)
        plt.close()
        return result

    # ----- stage: ipfx_fi_curve -----
    try:
        rates = sweep_table.loc[sweep_indexes, "avg_rate"].values
        result.update(run_feature_collection.fi_curve_fit(amps, rates))
    except Exception as e:
        _record_exception(result, "ipfx_fi_curve", e, specimen_label)
        plt.close()
        return result

    try:
        result["stimulus_name"] = data_set.sweepMetadata[0]['stim_dict']['description']
        result["sweeps_used"] = sweep_indexes
    except Exception as e:
        logger.debug("specimen=%s stage=metadata: %s", specimen_label, e)

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
    if t[0] <= t_0 <= t[-1]: f"Given time ({t_0}) is outside of time range ({t[0]}, {t[-1]})"
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
    try:
        start_time, duration, amplitude, start_idx, end_idx = get_stim_characteristics(i, t, test_pulse=test_pulse, start_epoch=start_epoch, end_epoch=end_epoch, test_pulse_length=test_pulse_length)
    except Exception:
        logger.debug("match_protocol: get_stim_characteristics failed", exc_info=True)
        return None
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


def diagnose_non_long_square(i, t, test_pulse=True, test_pulse_length=0.1):
    """Return (tag, detail_dict) explaining why `match_protocol` did not return
    'Long Square' for the given stimulus. Used purely for logging — does not
    affect analysis. `tag` values mirror the reject branches in
    `match_protocol` / `match_long_square_protocol`."""
    detail = {
        "len": int(len(i)) if i is not None else 0,
        "i_first": float(i[0]) if len(i) else float("nan"),
        "i_last": float(i[-1]) if len(i) else float("nan"),
        "i_min": float(np.nanmin(i)) if len(i) else float("nan"),
        "i_max": float(np.nanmax(i)) if len(i) else float("nan"),
        "i_abs_max": float(np.nanmax(np.abs(i))) if len(i) else float("nan"),
    }
    try:
        start_time, duration, amplitude, start_idx, end_idx = get_stim_characteristics(
            i, t, test_pulse=test_pulse, test_pulse_length=test_pulse_length,
        )
    except Exception as e:
        detail["err"] = f"{type(e).__name__}: {e}"
        return "stim_characteristics_raised", detail

    detail["start_time"] = None if start_time is None else float(start_time)
    detail["duration"] = None if duration is None else float(duration)
    detail["amplitude"] = None if amplitude is None else float(amplitude)
    if start_time is None:
        return "no_start_time", detail
    if duration is None:
        return "no_duration", detail
    if 0.1 <= duration <= 0.25:
        return "duration_mid_range", detail  # ramp path, currently returns None
    if duration < 0.1:
        return "duration_too_short", detail  # short square path, currently returns None
    # duration > 0.25 -> long-square branch
    di = np.diff(i)
    di_idx = np.flatnonzero(di)
    detail["num_transitions"] = int(len(di_idx))
    if len(di_idx) == 0:
        return "no_transitions", detail
    if len(di_idx) == 1:
        return "single_transition", detail
    if len(di_idx) > 6:
        try:
            y_data = i[start_idx:end_idx]
            x_data = t[start_idx:end_idx]
            slope, intercept, r_value, p_value, _ = scipy.stats.linregress(x_data, y_data)
            detail.update({
                "slope": float(slope),
                "p_value": float(p_value),
                "r_value": float(r_value),
            })
        except Exception as e:
            detail["regress_err"] = f"{type(e).__name__}: {e}"
        return "too_many_transitions_slope_fail", detail
    if len(i) and i[0] != 0:
        return "stim_nonzero_at_start", detail
    if len(i) and i[-1] != 0:
        return "stim_nonzero_at_end", detail
    return "unknown", detail
    

def match_long_square_protocol(i, t, start_idx, end_idx):
    #here we will do some analysis to determine if the stimulus is a long square, and if so, what the parameters are
    fs = 1/(t[1]  - t[0])

    di = np.diff(i)
    di_idx = np.flatnonzero(di)   # != 0

    if len(di_idx) == 0:
        #if there are no up/down transitions, then this is not a long square, or a zero-pulse long square but we will just have to accept that for now
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


def build_dataset_traces(folder, ids =None, ext="nwb", parallel=True):
    #this function will take a run_analysis function and return a dataset of traces
    files = glob_files(folder)[::-1]
    if ids is not None:
        files_check = [x.split("/dandi")[-1] in ids for x in files]
        files = np.array(files)[files_check]

    file_idx = np.arange(len(files))
    GLOBAL_STIM_NAMES.stim_inc = ['']
    GLOBAL_STIM_NAMES.stim_type = ['CurrentClamp', 'CC', 'IC', 'Current Clamp']
    if parallel == True:
        #Run in parallel
        parallel = 4
    else:
        parallel = 1
    #results = joblib.Parallel(n_jobs=parallel)(joblib.delayed(plot_data)(specimen_id, files) for specimen_id in file_idx)
    results = [plot_data(specimen_id, files) for specimen_id in file_idx]



def plot_data(specimen_id, file_list=None, target_amps=[-100, -20, 20, 100, 150, 250, 500, 1000], overwrite=False, save=True, stim_override=None) -> dict:
    result = {}
    if os.path.exists(f"{file_list[specimen_id]}.svg") and overwrite == False:
        print(f"skipping {file_list[specimen_id]} because it already exists")
        logging.debug(f"skipping {file_list[specimen_id]} because it already exists")
        return
    elif os.path.exists(f"{file_list[specimen_id]}.svg") and overwrite == True:
        #os.remove(f"{file_list[specimen_id]}.svg")
        pass
    else:
        pass

    if stim_override is not None:
        GLOBAL_STIM_NAMES.stim_inc = [stim_override]

    result["specimen_id"] = file_list[specimen_id]
    _, _, _,  data_set = loadNWB(file_list[specimen_id], return_obj=True)
    if data_set is None or len(data_set.dataY)<1:
        print(f"data_set is None for {file_list[specimen_id]}")
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
    if len(sweeps) < 1:
        return result
    #get the most common start and end times
    start_time = scipy.stats.mode(np.array(start_times))[0]
    end_time = scipy.stats.mode(np.array(end_times))[0]
    #index out the sweeps that have the most common start and end times
    idx_pass = np.where((np.array(start_times) == start_time) & (np.array(end_times) == end_time))[0]

    sweep_amp_filter = np.array(sweep_amp)[idx_pass]
    if len(sweep_amp_filter) < 1:
        return result
    #the sweeps closest to the target amp
    idx_targets = [np.argmin(np.abs(sweep_amp_filter - x)) for x in target_amps]

    idx_pass = idx_pass[idx_targets]

    #sweeps = np.array(sweeps, dtype=object)[idx_pass]

    sweeps = SweepSet(sweeps)
    #plot the sweeps
    fig, ax = plt.subplots(1, 1, figsize=(4,3))
    fi_s = []
    fi_i = []
    plotted = []
    sweep_ys = []
    sweep_xs = []
    for i, sweep in enumerate(sweeps.sweeps):
        idx_start = find_time_index(sweep.t, np.clip(start_time*0.8, 0.0, np.inf))
        idx_end = find_time_index(sweep.t, np.clip(end_time*1.4, end_time, sweep.t[-1]))
        if i in idx_pass and i not in plotted:
            plotted.append(i)
            ax.plot(sweep.t[idx_start:idx_end], sweep.v[idx_start:idx_end], label=f"sweep {i}", c='k', alpha=0.5)
            sweep_ys.append(sweep.v[idx_start:idx_end])
            sweep_xs.append(sweep.t[idx_start:idx_end])

        spikes = analyze_spike_times(sweep.t, sweep.v, sweep.i)
        fi_s.append(len(spikes) / (end_time - start_time))
        fi_i.append(sweep_amp[i])
        #ax.plot(sweep.t[idx_start:idx_end], sweep.v[idx_start:idx_end])



    #turn off the upper and right spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    #save the figure as a svg

    plt.savefig(f"{file_list[specimen_id]}.svg", bbox_inches='tight') if save==True else None

    fig2,ax = plt.subplots(figsize=(3,3))

    #if there is the same sweep_amp, average it
    fi_s = np.array(fi_s)
    fi_i = np.array(fi_i)

    #get duplicates of fi_i
    unique, counts = np.unique(fi_i, return_counts=True)
    duplicates = unique[counts > 1]
    for dup in duplicates:
        idx_dup = np.where(fi_i == dup)[0]
        fi_s[idx_dup] = np.mean(fi_s[idx_dup])
        fi_i[idx_dup] = np.mean(fi_i[idx_dup])
    #sort the fi_i and fi_s
    idx_sort = np.argsort(fi_i)
    fi_i = fi_i[idx_sort]
    fi_s = fi_s[idx_sort]
    #plot the fi curve


    plt.plot(fi_i, fi_s, c='k', marker='o')
    plt.xticks(np.arange(target_amps[0], target_amps[-1]+250, 250))
    plt.xlim( target_amps[0]-100, target_amps[-1]+100)
    plt.xlabel("Current (pA)")
    plt.ylabel("Firing Rate (Hz)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(f"{file_list[specimen_id]}_FI.svg", bbox_inches='tight') if save==True else None
    plt.close('all') if save==True else None



    print(f"saved {file_list[specimen_id]}.svg", end="\r") if save==True else None
    return {'fig_trace': fig, 'fi_i': fi_i, 'fi_s': fi_s, 'sweep_ys': sweep_ys, 'sweep_xs': sweep_xs, 'fi_fig':  fig2}
    

if __name__ == "__main__":
    freeze_support()
    #call main 
    main()

    

