###############################################################################
# This script will ensure that feature extraction is working properly
#
# Path: pyAPisolation/tests/test_feature_extractor.py
# We want to compare the output df to a known good df
# Unfortunately, the datafiles are not included in the repo, please contact me if you would like to run this test
# The known good df is saved in the test_data folder

import os
import pandas as pd
import numpy as np
from joblib import dump, load
from pyAPisolation.featureExtractor import batch_feature_extract, save_data_frames, analyze_spike_times, analyze_subthres
from pyAPisolation.patch_utils import load_protocols
from pyAPisolation.ipfx_df import save_subthres_data
import pyAPisolation.utils as utils
#from pyAPisolation.analysis import SpikeAnalysisModule
import glob
import pytest

COLS_TO_SKIP = ['Best Fit', 'Curve fit b1', #random / moving api
                 'foldername', 'protocol', #not a feature
                 ]


DEFAULT_DICT = {'filter': 0,
                 'start':0,
                 'end': 0,}

utils.DEBUG = True #enable debug mode for more verbose output

# ---- Demo data paths (included in repo under tests/test_data/) ----
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
DEMO_ABF_1 = os.path.join(TEST_DATA_DIR, 'demo_1.abf')
DEMO_ABF_2 = os.path.join(TEST_DATA_DIR, 'demo_2.abf')
DEMO_KNOWN_GOOD_PATH = os.path.join(TEST_DATA_DIR, 'demo_known_good_df.joblib')
HAS_DEMO_DATA = os.path.exists(DEMO_ABF_1) and os.path.exists(DEMO_ABF_2)


## ==== tests for the feature extractor w/ included demo data ====


def _compare_dataframes(df_known, df_test, cols_to_skip=None, threshold=0.01):
    """Compare two feature DataFrames column-wise using mean percentage error.
    Returns (passed, unequal_cols, diffs).
    """
    if cols_to_skip is None:
        cols_to_skip = COLS_TO_SKIP
    # Align by filename
    df_known = df_known.sort_values(by='filename').set_index('filename')
    df_test = df_test.sort_values(by='filename').set_index('filename')
    # Drop non-feature columns (only those that exist)
    drop_known = [c for c in cols_to_skip if c in df_known.columns]
    drop_test = [c for c in cols_to_skip if c in df_test.columns]
    df_known = df_known.drop(columns=drop_known)
    df_test = df_test.drop(columns=drop_test)

    if df_test.equals(df_known):
        return True, [], []

    unequal_cols = []
    diffs = []
    for col in df_known.columns:
        if col not in df_test.columns:
            unequal_cols.append(col)
            diffs.append(np.nan)
            continue
        if np.issubdtype(df_known[col].dtype, np.number):
            col1 = np.nan_to_num(df_known[col].to_numpy(), nan=-999)
            col2 = np.nan_to_num(df_test[col].to_numpy(), nan=-999)
            mpe = np.nanmean(np.abs(col1 - col2) / np.where(col1 == 0, 1, col1))
            if mpe >= threshold:
                unequal_cols.append(col)
                diffs.append(mpe)
        else:
            if not df_known[col].equals(df_test[col]):
                unequal_cols.append(col)
                diffs.append(np.nan)
    passed = len(diffs) == 0 or np.nanmean(diffs) < threshold
    return passed, unequal_cols, diffs


@pytest.mark.skipif(not HAS_DEMO_DATA, reason="Demo ABF files not found in tests/test_data/")
def test_demo_batch_feature_extract():
    """Run batch_feature_extract on the two demo ABF files and compare to a known-good reference.
    On first run the reference is bootstrapped and saved."""
    spike, feat_df, running = batch_feature_extract(TEST_DATA_DIR, DEFAULT_DICT, protocol_name='')

    # Basic sanity checks
    assert isinstance(feat_df, pd.DataFrame), "feat_df should be a DataFrame"
    assert len(feat_df) >= 1, "feat_df should have at least one row"
    assert isinstance(spike, pd.DataFrame), "spike should be a DataFrame"
    assert isinstance(running, pd.DataFrame), "running should be a DataFrame"

    # Bootstrap / regression comparison
    if not os.path.exists(DEMO_KNOWN_GOOD_PATH):
        dump(feat_df, DEMO_KNOWN_GOOD_PATH)
        print(f"Bootstrapped demo reference saved to {DEMO_KNOWN_GOOD_PATH}")
    else:
        df_known = load(DEMO_KNOWN_GOOD_PATH)
        passed, unequal_cols, diffs = _compare_dataframes(df_known, feat_df)
        assert passed, (
            f"Demo feature DataFrame does not match reference. "
            f"Unequal columns: {unequal_cols}, mean pct error: {np.nanmean(diffs)*100:.2f}%"
        )


@pytest.mark.skipif(not HAS_DEMO_DATA, reason="Demo ABF files not found in tests/test_data/")
def test_demo_analyze_spike_times():
    """Run analyze_spike_times on a demo ABF file."""
    result = analyze_spike_times(file=DEMO_ABF_1)
    # Should return without error; result may be empty if no spikes
    assert result is not None


@pytest.mark.skipif(not HAS_DEMO_DATA, reason="Demo ABF files not found in tests/test_data/")
def test_demo_analyze_subthres():
    """Run analyze_subthres on a demo ABF file."""
    dfs = analyze_subthres(
        file=DEMO_ABF_1, savfilter=0,
        start_sear=None, end_sear=None,
        subt_sweeps=None, time_after=50, bplot=False,
    )
    assert isinstance(dfs, tuple), "analyze_subthres should return a tuple"
    assert len(dfs) == 2, "analyze_subthres should return (df, avg)"
    assert isinstance(dfs[0], pd.DataFrame)
    assert isinstance(dfs[1], pd.DataFrame)


@pytest.mark.skipif(not HAS_DEMO_DATA, reason="Demo ABF files not found in tests/test_data/")
def test_demo_save_data_frames(tmp_path):
    """Run batch_feature_extract then save_data_frames to a temp directory."""
    spike, feat_df, running = batch_feature_extract(TEST_DATA_DIR, DEFAULT_DICT, protocol_name='')
    save_data_frames(spike, feat_df, running, root_fold=str(tmp_path))
    out_files = glob.glob(str(tmp_path + '/spike_count_*.xlsx'))
    assert len(out_files) >= 1, "save_data_frames should create an xlsx file"
    assert os.path.getsize(out_files[0]) > 0, "Output xlsx should be non-empty"


@pytest.mark.skipif(not HAS_DEMO_DATA, reason="Demo ABF files not found in tests/test_data/")
def test_demo_save_subthres_data(tmp_path):
    """Run analyze_subthres then save_subthres_data to a temp directory."""
    dfs = analyze_subthres(
        file=DEMO_ABF_1, savfilter=0,
        start_sear=None, end_sear=None,
        subt_sweeps=None, time_after=50, bplot=False,
    )
    save_subthres_data(dfs[1], dfs[0], root_fold=str(tmp_path))
    out_files = glob.glob(str(tmp_path / 'subthres_*.xlsx'))
    assert len(out_files) >= 1, "save_subthres_data should create an xlsx file"
    assert os.path.getsize(out_files[0]) > 0, "Output xlsx should be non-empty"


### === tests for larger feature extractor pipeline, have to be run manually since they require the data files, which are not included in the repo, please contact me if you would like to run these tests === ###
@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason="This test is not meant to run on GitHub Actions.")
def test_full_dataframe_save():
    # Run the feature extractor
    spike, feat_df, running = batch_feature_extract(os.path.expanduser('~/Dropbox/sara_cell_v2'), DEFAULT_DICT)
    #also test the save_data_frames function
    save_data_frames(spike, feat_df, running, root_fold=os.path.dirname(__file__))
    #checked manually later


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason="This test is not meant to run on GitHub Actions.")
def test_full_feature_extractor():
    # Load the known good df
    df = load(f'{os.path.dirname(__file__)}/test_data/known_good_df.joblib')

    # Run the feature extractor
    spike, feat_df, running = batch_feature_extract(os.path.expanduser('~/Dropbox/sara_cell_v2'), DEFAULT_DICT)
    
    #sort both by filename, index by filename
    df = df.sort_values(by='filename').set_index('filename')
    feat_df = feat_df.sort_values(by='filename').set_index('filename')

    # Drop the columns that are not tested against
    df = df.drop(columns=COLS_TO_SKIP)
    feat_df = feat_df.drop(columns=COLS_TO_SKIP)

    # Compare the two dataframes
    if feat_df.equals(df):
        print("Dataframes are equal")
        return
    unequal_cols = []
    diff = []
    # The output should be True
    #try to examine the dataframes to see what is different
    for col in df.columns:
            if col in feat_df.columns:
                #if its a numeric dtype
                if np.issubdtype(df[col].dtype, np.number):
                    col1 = df[col].to_numpy()
                    col2 = feat_df[col].to_numpy()
                    #nan_to_num the data to -999
                    col1 = np.nan_to_num(col1, nan=-999)
                    col2 = np.nan_to_num(col2, nan=-999)
                    #compute the mean perecentage error
                    mpe = np.nanmean(np.abs(col1 - col2)/col1)
                    if mpe < 0.01:
                        print(col, "is equal")
                    else:
                        print(f"WARNING: {col} is not equal; PLEASE CHECK, mean percent error is {mpe*100}")
                        unequal_cols.append(col)
                        diff.append(mpe)
                else:
                    #check if they are equal using pandas
                    if df[col].equals(feat_df[col]):
                        print(col, "is equal")
                    else:
                        print(f"WARNING: {col} is not equal; PLEASE CHECK")
                        unequal_cols.append(col)
                        diff.append(np.nan)
            else:
                print(f"WARNING: {col} is not in test_feature_extractor")
    #check the error threshold
    if np.nanmean(diff) < 0.01:
        print("Dataframes are equal")
        return
    else:
        #write to excel for manual inspection
        with pd.ExcelWriter('diffs.xlsx') as writer:
            df.to_excel(writer, sheet_name='known_good_df')
            feat_df.to_excel(writer, sheet_name='feat_df')
            #also write the differences
            diff_df = pd.DataFrame({'col': unequal_cols, 'diff': diff})
            diff_df.to_excel(writer, sheet_name='diffs')

            df_unequal = df[unequal_cols]
            feat_df_unequal = feat_df[unequal_cols]

            df_unequal.join(feat_df_unequal, lsuffix='_known_good', rsuffix='_feat').to_excel(writer, sheet_name='unequal_cols')


        assert False, f"Dataframes are not equal, mean percent error is {np.nanmean(diff)*100}"
    print("Dataframes are not equal, but mean percent error is below threshold, please check diffs.xlsx for details")


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason="This test is not meant to run on GitHub Actions.")
def test_full_analyze_funcs():
    files = glob.glob(os.path.expanduser('~/Dropbox/sara_cell_v2') + '/**/*.abf', recursive=True)
    #load the protocols
    #try to load the protocols
    spike_times = analyze_spike_times(file=files[-1])
    print(spike_times)


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason="This test is not meant to run on GitHub Actions.")
def test_full_subthreshold_funcs():
    #load a file
    files = glob.glob(os.path.expanduser('./data/') + '/**/*.abf', recursive=True)
    
    
    dfs = analyze_subthres(file=files[0],  savfilter=0, start_sear=None, end_sear=None, subt_sweeps=None, time_after=50, bplot=False)
    save_subthres_data(dfs[1], dfs[0], root_fold=os.path.dirname(__file__))


@pytest.mark.skipif(os.getenv('GITHUB_ACTIONS') == 'true', reason="This test is not meant to run on GitHub Actions.")
def test_full_modular_analysis():
    #we need to make sure the modular feature analysis is working:
    # Load the known good df
    df = load(f'{os.path.dirname(__file__)}/test_data/known_good_df.joblib')
    files = glob.glob(os.path.expanduser('~/Dropbox/sara_cell_v2') + '/**/*.abf', recursive=True)
    # Run the feature extractor
    #spike, feat_df, running = batch_feature_extract(os.path.expanduser('~/Dropbox/sara_cell_v2'), DEFAULT_DICT)

    # Initialize the SpikeAnalysisModule
    spike_analysis_module = SpikeAnalysisModule()
    #run one file
    res = spike_analysis_module.analyze(file=files[-1])
    dict_parallel = DEFAULT_DICT.copy()
    dict_parallel['n_jobs'] = 4
    spike_analysis_module.run_batch_analysis(files, param_dict=dict_parallel)
    # Get the results
    results = spike_analysis_module.get_results()
    # Test the analyze_subthres function


if __name__ == '__main__':
    pytest.main([__file__])
    