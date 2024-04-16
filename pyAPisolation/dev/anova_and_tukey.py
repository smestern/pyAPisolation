# This script is used to perform ANOVA and Tukey's HSD test on the data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.stats import oneway, multicomp
from scipy.stats import f_oneway
import copy
from sklearn.impute import SimpleImputer, KNNImputer

frame = pd.read_excel('C:\\Users\\SMest\\Dropbox\\sara_cell_v2\\spike_count_1712533162.9597082filtered.xlsx')
frame['Cell type'] = frame['foldername.1'].astype('category').map(lambda x: ''.join(x.split('\\')[:-1]))
labels = frame['Cell type'].to_numpy()  

arr = np.nan_to_num(np.array(frame.select_dtypes(include=np.number).values, dtype=float), nan=np.nan, posinf=np.nan, neginf=np.nan)
feat_labels = frame.select_dtypes(include=np.number).columns.to_list()

imputer = KNNImputer()
arr = imputer.fit_transform(arr)

arr = np.hsplit(arr, arr.shape[1])

test_res = []
tukey_res = []

for i, col in enumerate(arr):
    print(f"{i}/{len(arr)}")
    anova_res = oneway.anova_oneway(col[:,0], groups=labels, use_var='equal', welch_correction=True, trim_frac=0)
    test_res.append(anova_res)
    if anova_res.pvalue < 0.05:
        print('found a significant feature {}'.format(feat_labels[i]))
        tukey = multicomp.pairwise_tukeyhsd(col[:,0], labels, alpha=0.05)
        tukey_res.append(copy.deepcopy(tukey))
        
    else:
        #continue
        tukey_res.append(None)


# output the results table
#find the index of the first significant result
idx = [i for i, x in enumerate(test_res) if x.pvalue < 0.05][0]

df_full = pd.read_html(tukey_res[idx].summary().as_html())[0]
df_full.set_index(['group1', 'group2'], inplace=True)
df_full = df_full[['reject']]
for res, label in zip(tukey_res[0:], feat_labels[1:]):
    if res is not None and 'PCA' not in label:
        df = pd.read_html(res.summary().as_html())[0]
        df.set_index(['group1', 'group2'], inplace=True)
        df_full = df_full.join(df['reject'], rsuffix=f"f_{label}")

df_full.to_csv('tukey_results.csv')

print(anova_res)

