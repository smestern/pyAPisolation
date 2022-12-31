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
import ipfx
import ipfx.script_utils as su
from ipfx.stimulus import StimulusOntology
import allensdk.core.json_utilities as ju
from ipfx.bin import run_feature_collection
from ipfx import script_utils as su
#import ephys dataset
from ipfx.dataset.create import create_ephys_data_set
import copy
import joblib
# Import custom functions4 cc current injection action potential
from pyAPisolation import patch_utils
#GLOBALS
_ONTOLOGY = ju.read(StimulusOntology.DEFAULT_STIMULUS_ONTOLOGY_FILE)

def glob_files(folder, ext="nwb"):
    #this function will take a folder and a file extension, and return a list of files
    # with that extension in that folder
    return glob.glob(folder + "/**/*." + ext, recursive=True)


def run_analysis(folder, backend="ipfx", outfile='out.csv', ext="nwb", parallel=False):

    files = glob_files(folder)
    file_idx = np.arange(len(files))
     
    if backend == "ipfx":
        # Use ipfx to extract features
        temp_ontology = get_stimulus_protocols(files)
        get_data_partial = partial(data_for_specimen_id,
                                passed_only=False,
                                data_source='filesystem',
                                ontology=StimulusOntology(temp_ontology),
                                file_list=files)
        #if parallel == True:
            # Run in parallel
            #parallel = joblib.cpu_count()
        results = list(map(get_data_partial, file_idx))#joblib.Parallel(n_jobs=parallel)(joblib.delayed(get_data_partial)(specimen_id) for specimen_id in files)
        # Save results
        #with open(outfile, 'w') as f:
            #json.dump(results, f)

    elif backend == "custom":
        raise(NotImplementedError)
        # Use custom backend to extract features
        results = []
        for f in files:
            # Extract features from each file
            result = feature_extraction.extract_features(f)
            results.append(result)
        # Save results
        df = pd.DataFrame(results)
        df.to_csv(outfile, index=False)

    results = pd.concat([pd.DataFrame(r) for r in results])
    return results


def main():
    #main function to be called when running this script
    # Handle command line arguments
    parser = argparse.ArgumentParser(description='Build database for web visualization')
    parser.add_argument('folder', type=str, help='Folder containing ABF or NWB files')
    parser.add_argument('--backend', type=str, default="ipfx", help='Backend to use for feature extraction')
    parser.add_argument('--outfile', type=str, default="out.csv", help='Output file name')
    args = parser.parse_args()
    # Run analysis
    run_analysis(args.folder, args.backend, args.outfile)


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

    for i, f in enumerate(files):
        # Extract features from each file
        try:
            data_set = su.dataset_for_specimen_id(i, 'filesystem', StimulusOntology(_ONTOLOGY), files)
            sweep_table = data_set.sweep_table.loc[data_set.sweep_table['clamp_mode']=='CurrentClamp']
            unique_stim = sweep_table['stimulus_code'].unique()
            for stim in unique_stim:
                sweep_idx = sweep_table[sweep_table['stimulus_code']==stim].index[0]
                sweep = data_set.sweep(sweep_idx)
                i = sweep.i
                t = sweep.t
                stim_protocol = match_protocol(i, t)
                if stim_protocol is not None:
                    #add stim_protocol to ontology
                    _ONTOLOGY.append([['code', stim, stim_protocol], [ 'name', stim,  stim_protocol]])
        except:
            pass

    return copy.deepcopy(_ONTOLOGY)

def match_protocol(i, t, test_pulse=True, start_epoch=None, end_epoch=None, test_pulse_length=0.1):
    #this function will take a stimulus and return the stimulus protocol at least it will try
    #first we will try to match the stimulus protocol to a known protocol
    
    start_time, duration, amplitude, start_idx, end_idx = get_stim_characteristics(i, t, test_pulse=test_pulse, start_epoch=start_epoch, end_epoch=end_epoch, test_pulse_length=test_pulse_length)
    if start_time is None:
        #if we can't find the start time, then we can't identify the stimulus protocol
        return None
    if duration > 0.5:
        #if the stimulus is longer than 500ms, then it is probably a long square
        return match_long_square_protocol(i, t)
    elif duration < 0.1:
        #if the stimulus is less than 100ms, then it is probably a short square
        return match_short_square_protocol(i, t)
    else:
        #check if ramp
        return match_ramp_protocol(i, t)
    

def match_long_square_protocol(i, t):
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
    if len(di_idx) > 5:
        #if there are more than 5 up/down transitions, then this is not a long square
        return None


    #ensure that the stim starts at 0, and ends at 0
    if i[0] != 0:
        return None
    if i[-1] != 0:
        return None
    

    return "Long Square"

def match_short_square_protocol(stimulus_protocol, ontology):
    pass

def match_ramp_protocol(stimulus_protocol, ontology):
    pass



#IPFX functions
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
    #we are assuming that the test pulse is within the first 100ms of the stimulus
    #TODO make this more robust
    if (di_idx[1]) < test_pulse_length*fs: # skip the first up/down (test pulse) if present
        start_idx_idx = 2
    else:
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


def data_for_specimen_id(specimen_id, passed_only, data_source, ontology, file_list=None):
    #this is a clone of the function in ipfx/bin/run_feature_collection.py,
    # here we are gonna try to use it to handle data that may not be in an NWB format IPFX can handle
    data_set = su.dataset_for_specimen_id(specimen_id, data_source, ontology, file_list)
        

    try:
        lsq_sweep_numbers = su.categorize_iclamp_sweeps(data_set, ontology.long_square_names)
        ssq_sweep_numbers = su.categorize_iclamp_sweeps(data_set, ontology.short_square_names)
        ramp_sweep_numbers = su.categorize_iclamp_sweeps(data_set, ontology.ramp_names)
    except Exception as detail:
        logging.warn("Exception when processing specimen {:d}".format(specimen_id))
        logging.warn(detail)
#         return {"error": {"type": "sweep_table", "details": traceback.format_exc(limit=1)}}
        return {}
    try:
        result = run_feature_collection.extract_features(data_set, ramp_sweep_numbers.tolist(), ssq_sweep_numbers.tolist(), lsq_sweep_numbers.tolist())
    except Exception as detail:
        logging.warn("Exception when processing specimen {:d}".format(specimen_id))
        logging.warn(detail)
#         return {"error": {"type": "processing", "details": traceback.format_exc(limit=1)}}
        return {}

    result["specimen_id"] = specimen_id
    return result

from dandi.dandiapi import DandiAPIClient
from dandi.download import download as dandi_download
from collections import defaultdict
#dandi functions


def build_dandiset_df():
    client = DandiAPIClient()

    dandisets = list(client.get_dandisets())

    species_replacement = {
        "Mus musculus - House mouse": "House mouse",
        "Rattus norvegicus - Norway rat": "Rat",
        "Brown rat": "Rat",
        "Rat; norway rat; rats; brown rat": "Rat",
        "Homo sapiens - Human": "Human",
        "Drosophila melanogaster - Fruit fly": "Fruit fly",
    }

    neurodata_type_map = dict(
        ecephys=["LFP", "Units", "ElectricalSeries"],
        ophys=["PlaneSegmentation", "TwoPhotonSeries", "ImageSegmentation"],
        icephys=["PatchClampSeries", "VoltageClampSeries", "CurrentClampSeries"],
    )

    def is_nwb(metadata):
        return any(
            x['identifier'] == 'RRID:SCR_015242'
            for x in metadata['assetsSummary'].get('dataStandard', {})
        )

    data = defaultdict(list)
    for dandiset in dandisets:
        identifier = dandiset.identifier
        metadata = dandiset.get_raw_metadata()
        if not is_nwb(metadata) or not dandiset.draft_version.size:
            continue
        data["identifier"].append(identifier)
        data["created"].append(dandiset.created)
        data["size"].append(dandiset.draft_version.size)
        if "species" in metadata["assetsSummary"] and len(metadata["assetsSummary"]["species"]):
            data["species"].append(metadata["assetsSummary"]["species"][0]["name"])
        else:
            data["species"].append(np.nan)
        
        
        for modality, ndtypes in neurodata_type_map.items():
            data[modality].append(
                any(x in ndtypes for x in metadata["assetsSummary"]["variableMeasured"])
            )
        
        if "numberOfSubjects" in metadata["assetsSummary"]:
            data["numberOfSubjects"].append(metadata["assetsSummary"]["numberOfSubjects"])
        else:
            data["numberOfSubjects"].append(np.nan)
        
    df = pd.DataFrame.from_dict(data)

    for key, val in species_replacement.items():
        df["species"] = df["species"].replace(key, val)
    return df

def analyze_dandiset(code, cache_dir=None):
    df_dandiset = run_analysis(cache_dir+'/'+code)
    return df_dandiset
    

def download_dandiset(code=None, save_dir=None, overwrite=False):
    client = DandiAPIClient()
    dandiset = client.get_dandiset(code)
    if save_dir is None:
        save_dir = os.getcwd()
    if os.path.exists(save_dir+'/'+code) and overwrite==False:
        return
    dandi_download(dandiset.api_url, save_dir)
    
dandisets_to_skip = ['000012', '000013', '000005']
def run_analyze_dandiset():
    dandi_df = build_dandiset_df()
    filtered_df = dandi_df[dandi_df["icephys"] == True]
    filtered_df = filtered_df[filtered_df["numberOfSubjects"] > 2]
    for row in filtered_df.iterrows():
        if row[1]["identifier"] in dandisets_to_skip:
            continue
        download_dandiset(row[1]["identifier"], save_dir='/media/smestern/Expansion/dandi', overwrite=False)
        df_dandiset = analyze_dandiset(row[1]["identifier"],cache_dir='/media/smestern/Expansion/dandi/')
        df_dandiset["dandiset"] = row[1]["identifier"]
        df_dandiset["created"] = row[1]["created"]
        df_dandiset["species"] = row[1]["species"]
        df_dandiset.to_csv('/media/smestern/Expansion/dandi/'+row[1]["identifier"]+'.csv')


if __name__ == "__main__":
    #run_analysis('/media/smestern/Expansion/dandi/000020', backend="ipfx", outfile='out.csv', ext="nwb", parallel=1)
    run_analyze_dandiset()

