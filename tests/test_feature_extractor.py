###############################################################################
# This script will ensure that feature extraction is working properly
#
# Path: pyAPisolation/tests/test_feature_extractor.py
# We want to compare the output df to a known good df
# Unfortunately, the datafiles are not included in the repo, please contact me if you would like to run this test
# The known good df is saved in the test_data folder

import os
import shutil
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
DEMO_ABFS = sorted(glob.glob(os.path.join(TEST_DATA_DIR, 'demo_*.abf')))
DEMO_ABF_IDS = [os.path.basename(p) for p in DEMO_ABFS]
# Back-compat single-file aliases (kept for any external callers)
DEMO_ABF_1 = DEMO_ABFS[0] if len(DEMO_ABFS) > 0 else os.path.join(TEST_DATA_DIR, 'demo_1.abf')
DEMO_ABF_2 = DEMO_ABFS[1] if len(DEMO_ABFS) > 1 else os.path.join(TEST_DATA_DIR, 'demo_2.abf')
DEMO_ABF_3 = DEMO_ABFS[2] if len(DEMO_ABFS) > 2 else os.path.join(TEST_DATA_DIR, 'demo_3.abf')
DEMO_KNOWN_GOOD_PATH = os.path.join(TEST_DATA_DIR, 'demo_known_good_df.joblib')
HAS_DEMO_DATA = len(DEMO_ABFS) >= 1

#
ROUTE_PYTEST = False #set to False to skip the tests that require the demo data, which is not included in the repo, please contact me if you would like to run these tests


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
        if pd.api.types.is_numeric_dtype(df_known[col]):
            col1 = np.nan_to_num(df_known[col].to_numpy(dtype=float), nan=-999)
            col2 = np.nan_to_num(df_test[col].to_numpy(dtype=float), nan=-999)
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
    """Run batch_feature_extract over all demo ABFs and compare to a known-good reference.
    On first run the reference is bootstrapped and saved."""
    spike, feat_df, running = batch_feature_extract(TEST_DATA_DIR, DEFAULT_DICT, protocol_name='')

    # Basic sanity checks
    assert isinstance(feat_df, pd.DataFrame), "feat_df should be a DataFrame"
    assert len(feat_df) >= 1, "feat_df should have at least one row"
    assert isinstance(spike, pd.DataFrame), "spike should be a DataFrame"
    assert isinstance(running, pd.DataFrame), "running should be a DataFrame"

    # Every demo ABF should be represented in the output (catches silently-skipped files).
    assert 'filename' in feat_df.columns, "feat_df must have a 'filename' column"
    out_names = set(feat_df['filename'].astype(str).tolist())
    for abf in DEMO_ABFS:
        stem = os.path.splitext(os.path.basename(abf))[0]
        assert any(stem in n for n in out_names), (
            f"Demo file {stem} missing from batch_feature_extract output. Got: {sorted(out_names)}"
        )

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
@pytest.mark.parametrize("abf_path", DEMO_ABFS, ids=DEMO_ABF_IDS)
def test_demo_batch_single_file(tmp_path, abf_path):
    """Run batch_feature_extract on a single demo ABF (isolated in a temp dir) and
    compare its row against the aggregate known-good reference. Pinpoints which file
    regressed when test_demo_batch_feature_extract fails."""
    # Stage the single ABF in an isolated directory so batch sees only this file.
    staged = tmp_path / os.path.basename(abf_path)
    shutil.copy(abf_path, staged)

    spike, feat_df, running = batch_feature_extract(str(tmp_path), DEFAULT_DICT, protocol_name='')

    assert isinstance(feat_df, pd.DataFrame)
    assert len(feat_df) == 1, (
        f"Expected exactly one feature row for {os.path.basename(abf_path)}, got {len(feat_df)}"
    )
    assert 'filename' in feat_df.columns

    if not os.path.exists(DEMO_KNOWN_GOOD_PATH):
        pytest.skip("Reference DEMO_KNOWN_GOOD_PATH not yet bootstrapped")

    df_known_full = load(DEMO_KNOWN_GOOD_PATH)
    stem = os.path.splitext(os.path.basename(abf_path))[0]
    mask = df_known_full['filename'].astype(str).str.contains(stem, regex=False)
    df_known = df_known_full[mask]
    assert len(df_known) == 1, (
        f"Reference does not contain exactly one row for {stem} (found {len(df_known)}); "
        f"reference may be out of date — delete {DEMO_KNOWN_GOOD_PATH} to rebootstrap."
    )

    # Restrict to columns present in both: single-file batches legitimately omit
    # sweep-indexed columns for sweeps the other demo files contributed.
    shared_cols = [c for c in df_known.columns if c in feat_df.columns]
    df_known = df_known[shared_cols]
    feat_df_cmp = feat_df[shared_cols]

    passed, unequal_cols, diffs = _compare_dataframes(df_known, feat_df_cmp)
    assert passed, (
        f"Single-file regression for {os.path.basename(abf_path)} failed. "
        f"Unequal columns: {unequal_cols}, mean pct error: {np.nanmean(diffs)*100:.2f}%"
    )


@pytest.mark.skipif(not HAS_DEMO_DATA, reason="Demo ABF files not found in tests/test_data/")
@pytest.mark.parametrize("abf_path", DEMO_ABFS, ids=DEMO_ABF_IDS)
def test_demo_analyze_spike_times(abf_path):
    """Run analyze_spike_times on each demo ABF file."""
    result = analyze_spike_times(file=abf_path)
    # analyze_template returns a numpy array (possibly empty if no spikes / no 'sweep Number').
    assert isinstance(result, np.ndarray), (
        f"analyze_spike_times should return ndarray, got {type(result).__name__}"
    )
    if result.size > 0:
        assert result.ndim == 2, f"Non-empty result should be 2D, got ndim={result.ndim}"


@pytest.mark.skipif(not HAS_DEMO_DATA, reason="Demo ABF files not found in tests/test_data/")
@pytest.mark.parametrize("abf_path", DEMO_ABFS, ids=DEMO_ABF_IDS)
def test_demo_analyze_subthres(abf_path):
    """Run analyze_subthres on each demo ABF file."""
    dfs = analyze_subthres(
        file=abf_path, savfilter=0,
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
    out_files = glob.glob(os.path.join(str(tmp_path), 'spike_count_*.xlsx'))
    assert len(out_files) >= 1, "save_data_frames should create an xlsx file"
    assert os.path.getsize(out_files[0]) > 0, "Output xlsx should be non-empty"


@pytest.mark.skipif(not HAS_DEMO_DATA, reason="Demo ABF files not found in tests/test_data/")
@pytest.mark.parametrize("abf_path", DEMO_ABFS, ids=DEMO_ABF_IDS)
def test_demo_save_subthres_data(tmp_path, abf_path):
    """Run analyze_subthres then save_subthres_data to a temp directory for each demo ABF."""
    dfs = analyze_subthres(
        file=abf_path, savfilter=0,
        start_sear=None, end_sear=None,
        subt_sweeps=None, time_after=50, bplot=False,
    )
    save_subthres_data(dfs[1], dfs[0], root_fold=str(tmp_path))
    out_files = glob.glob(os.path.join(str(tmp_path), 'subthres_*.xlsx'))
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
    files = glob.glob(os.path.dirname(__file__) + '/**/*.abf', recursive=True)
    
    
    dfs = [analyze_subthres(file=files[x],  savfilter=0, start_sear=None, end_sear=None, subt_sweeps=None, time_after=50, bplot=False) for x in range(len(files))]
    main = pd.concat([dfs[x][0] for x in range(len(dfs))])
    avg = pd.concat([dfs[x][1] for x in range(len(dfs))])
    save_subthres_data(avg, main, root_fold=os.path.dirname(__file__))


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
    if ROUTE_PYTEST:
        pytest.main([__file__])
    else:
        print("Running full feature extractor test (not recommended for regular use, requires demo data files)...")
        # test_full_dataframe_save()
        # test_full_feature_extractor()
        # test_full_analyze_funcs()
        test_full_subthreshold_funcs()
    
    