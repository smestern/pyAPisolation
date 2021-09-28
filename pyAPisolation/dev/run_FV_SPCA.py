print("Loading...")
import sys
import numpy as np
from numpy import genfromtxt
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from sklearn.preprocessing import minmax_scale, StandardScaler
from sklearn.decomposition import SparsePCA
from sklearn.impute import SimpleImputer
import glob

#glob the fv csv's 
fv_files = glob.glob("*.csv")

fv_dict = {}

#load the used ids
ids =  [x.split('.')[0] for x in pd.read_csv("ids.csv", header=None, names=['ids'])['ids'].to_numpy()]

#load the cluster ids
cell_type_df = pd.read_csv("C:\\Users\\SMest\\Documents\\clustering-data\\MARM_PVN_IC1\\spike_count_sort_out.csv")
file_names = cell_type_df['filename'].to_numpy()
cell_type_df = cell_type_df.set_index('filename')
cell_type_label = cell_type_df['cell_label'].to_numpy()

rows_to_use = np.in1d(ids, file_names)
labels = cell_type_label[np.argwhere(file_names==ids)]


#L
df_out = pd.DataFrame(data=np.array(ids)[rows_to_use].reshape(-1,1), columns=['filename'], index=np.array(ids)[rows_to_use])

#df_out['label'] = labels_out

labels_idx = np.hstack([np.ravel(np.argwhere(x==np.array(ids)[rows_to_use])) for x in file_names])
labels_out = cell_type_label[labels_idx]


#the sparse method
spca = SparsePCA(n_components=3, max_iter=4000, n_jobs=-1)
#imputer 
impute = SimpleImputer()
#scale
scale = StandardScaler()

for f in fv_files:
    
    data = np.genfromtxt(f, delimiter=',')
    key = os.path.basename(f).split('.')[0]
    
    print(f"Reducing {key}")
    if ('id' in key) or ('neuron' in key):
        continue
    else:
        data = data[rows_to_use, :]
        data = impute.fit_transform(data)
        data = scale.fit_transform(data)
        data_reduced = spca.fit_transform(data)
        fv_dict[key] = data_reduced
    #np.savetxt(f"{key}_spca.csv", data_reduced, delimiter=',', fmt='%.4f')




for key, val in fv_dict.items():
    df_out[f'{key}_0'] = val[:,0]
    df_out[f'{key}_1'] = val[:,1]
    df_out[f'{key}_2'] = val[:,2]



df_out2 = df_out.join(cell_type_df, on='filename', how='right', lsuffix='_left', rsuffix='_right')


df_out2.to_csv('fv_SPCA.csv')

