###############################################################################
# This script will ensure that feature extration is working properly
#
# Path: pyAPisolation/tests/test_feature_extractor.py
# We want to compare the output df to a known good df
# Unfortunately, the datafiles are not included in the repo, please contact me if you would like to run this test
# The known good df is saved in the test_data folder

import os
import pandas as pd
import numpy as np
from joblib import dump, load
from pyAPisolation.feature_extractor import folder_feature_extract, save_data_frames, default_dict
from pyAPisolation.patch_utils import load_protocols

def test_feature_extractor():
    # Load the known good df
    df = load(f'{os.path.dirname(__file__)}/test_data/known_good_df.joblib')

    # Run the feature extractor
    _, feat_df,_ = folder_feature_extract(f'/media/smestern/Expansion/IC1 Files_220106_183 cells', default_dict)
    
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
        assert False, f"Dataframes are not equal, mean percent error is {np.nanmean(diff)*100}"

if __name__ == '__main__':
    test_feature_extractor()