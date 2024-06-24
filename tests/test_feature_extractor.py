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
from pyAPisolation.featureExtractor import batch_feature_extract, save_data_frames, default_dict
from pyAPisolation.patch_utils import load_protocols


COLS_TO_SKIP = ['Best Fit', 'Curve fit b1', #random / moving api
                 'foldername', 'protocol', #not a feature
                 ]


def test_dataframe_save():
    # Run the feature extractor
    spike, feat_df, running = batch_feature_extract(os.path.expanduser('~/Dropbox/sara_cell_v2'), default_dict)
    #also test the save_data_frames function
    save_data_frames(spike, feat_df, running, root_fold=os.path.dirname(__file__))
    #checked manually later

def test_feature_extractor():
    # Load the known good df
    df = load(f'{os.path.dirname(__file__)}/test_data/known_good_df.joblib')

    # Run the feature extractor
    spike, feat_df, running = batch_feature_extract(os.path.expanduser('~/Dropbox/sara_cell_v2'), default_dict)
    
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



if __name__ == '__main__':
    test_dataframe_save()
    test_feature_extractor()