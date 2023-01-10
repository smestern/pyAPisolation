
#for heroku deployment we have to force install some packages here
import sys
import subprocess
packages = ['dandi', 'dash-table',]
# implement pip as a subprocess:
for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])



from dandi.dandiapi import DandiAPIClient
from dandi.download import download as dandi_download
from collections import defaultdict
#dandi functions
# os sys imports
from pyAPisolation.loadNWB import loadFile, loadNWB, GLOBAL_STIM_NAMES
from pyAPisolation.abf_featureextractor import *
from pyAPisolation.patch_ml import *
import os

import glob
import argparse
import scipy.stats
# dash / plotly imports
import dash
from dash.dependencies import Input, Output, State
import dash_table
import dash_core_components as dcc
import dash_bootstrap_components as dbc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
# data science imports
import pandas as pd
import numpy as np
import pyabf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from .build_database import run_analysis, parse_long_pulse_from_dataset
from .dash_folder_app import live_data_viz, GLOBAL_VARS
from sklearn.ensemble import IsolationForest

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

        data['keywords'].append([x.lower() for x in metadata.get("keywords", [])])
    df = pd.DataFrame.from_dict(data)

    for key, val in species_replacement.items():
        df["species"] = df["species"].replace(key, val)
    return df

def analyze_dandiset(code, cache_dir=None):
    df_dandiset = run_analysis(cache_dir+'/'+code)
    return df_dandiset
    
def filter_dandiset_df(row, species=None, modality=None, keywords=[], method='or'):
    flags = []
    if species is not None:
        flags.append(row["species"] == species)
    if modality is not None:
        flags.append(row[modality] == True)
    if len(keywords) > 0:
        flags.append(np.any(np.ravel([[x in j for j in row["keywords"]] for x in keywords])))
    if method == 'or':
        return any(flags)
    elif method == 'and':
        return all(flags)
    else:
        raise ValueError("method must be 'or' or 'and'")


def download_dandiset(code=None, save_dir=None, overwrite=False):
    client = DandiAPIClient()
    dandiset = client.get_dandiset(code)
    if save_dir is None:
        save_dir = os.getcwd()
    if os.path.exists(save_dir+'/'+code) and overwrite==False:
        return
    dandi_download(dandiset.api_url, save_dir)
    

def quick_qc(df):
    #this is a quick qc function to check if the data is good, specifically for the output of the analyze_dandiset function
    #this is not a full qc function, but it is a good start
    #check if the input resistence is reasonable
    df = df[df['input_resistance'] > 0]
    df = df[df['input_resistance'] < 1e8]

    #check if the resting potential is reasonable
    #df = df[df['resting_potential'] > -1e3]
    #df = df[df['resting_potential'] < 1e3]

    #check the ap_1_threshold_v_0_long_square
    df = df[df['ap_1_threshold_v_0_long_square'] > -100]
    df = df[df['ap_1_threshold_v_0_long_square'] < 100]

    #perform basic outliers checks
    df_num = df.select_dtypes(include=np.number).fillna(0)
    if_outlier = IsolationForest(random_state=0).fit_predict(df_num)
    #df = df[if_outlier == 1]
    return df


dandisets_to_skip = ['000012', '000013', '000005', '000117', '000168', 
'000362', #appears to be some lfp or something
 '000293', #Superseeded by 000297
  '000292',
  '000341'] #Superseeded by 000297
dandisets_to_include = ['000008', '000035']
def run_analyze_dandiset():
    dandi_df = build_dandiset_df()
    filtered_df = dandi_df[dandi_df.apply(lambda x: filter_dandiset_df(x, modality='icephys', keywords=['intracellular', 'patch'], method='or'), axis=1)]
    
    print(len(filtered_df))
    #glob the csv files
    csv_files = glob.glob('/media/smestern/Expansion/dandi/*.csv')
    csv_files = [x.split('/')[-1].split('.')[0] for x in csv_files]
    filtered_df = filtered_df[~filtered_df["identifier"].isin(csv_files)]


    for row in filtered_df.iterrows():
        print(f"Downloading {row[1]['identifier']}")
        if row[1]["identifier"] in dandisets_to_skip:
            continue
        download_dandiset(row[1]["identifier"], save_dir='/media/smestern/Expansion/dandi', overwrite=False)
        df_dandiset = analyze_dandiset(row[1]["identifier"],cache_dir='/media/smestern/Expansion/dandi/')
        df_dandiset["dandiset"] = row[1]["identifier"]
        df_dandiset["created"] = row[1]["created"]
        df_dandiset["species"] = row[1]["species"]
        df_dandiset.to_csv('/media/smestern/Expansion/dandi/'+row[1]["identifier"]+'.csv')

def run_merge_dandiset():
    from sklearn.preprocessing import StandardScaler
    import umap
    from sklearn.impute import SimpleImputer
    from sklearn.mixture import GaussianMixture
    csv_files = glob.glob('/media/smestern/Expansion/dandi/*.csv')
    csv_files = [x.split('/')[-1].split('.')[0] for x in csv_files]
    dfs = []
    dataset_numeric = []
    for code in csv_files:
        temp_df = pd.read_csv('/media/smestern/Expansion/dandi/'+code+'.csv', index_col=0)
        temp_df = temp_df.dropna(axis=0, how='all', subset=temp_df.columns[:-3])
        temp_df.rename(columns={'dandiset': 'dandiset label', 'species label': 'species'}, inplace=True)
        
        dfs.append(temp_df)
        

    dfs = pd.concat(dfs)
    #remap the dandiset label so that its a string
    dfs['dandiset label'] = dfs['dandiset label'].apply(lambda x: '000000'[:6-len(str(x))]+str(x))
    idxs = []
    columns = []
    for code in dfs['dandiset label'].unique():
        temp_df = dfs.loc[dfs['dandiset label'] == code]
        data_num = temp_df.select_dtypes(include=np.number).dropna(axis=1, how='all')
        temp_data_num = data_num.copy()
        if data_num.empty or len(data_num.columns) < 10:
            continue
        idxs.append(data_num.index.values)
        scale = StandardScaler()
        data_num = scale.fit_transform(data_num)
        impute = SimpleImputer()
        data_num = impute.fit_transform(data_num)

        data_num = pd.DataFrame(data_num, columns=temp_data_num.columns)
        dataset_numeric.append(data_num)
    dataset_numeric = pd.concat(dataset_numeric, axis=0)
    dfs = dfs.loc[np.concatenate(idxs)]
    #embed the data
    reducer = umap.UMAP(n_neighbors=150)
    embedding = reducer.fit_transform(dataset_numeric.fillna(0))
    dfs['umap X'] = embedding[:,0]
    dfs['umap Y'] = embedding[:,1]

    #generate some labels for fun
    gmm = GaussianMixture(n_components=15)
    gmm.fit(dataset_numeric.fillna(0))
    dfs['GMM cluster label'] = gmm.predict(dataset_numeric.fillna(0))
    
    dfs = quick_qc(dfs)

    dfs.to_csv('/media/smestern/Expansion/dandi/all.csv')



import fsspec
import pynwb
import h5py
from fsspec.implementations.cached import CachingFileSystem


class dandi_data_viz(live_data_viz):
    def __init__(self, database_file=None):
        super(dandi_data_viz, self).__init__(database_file=database_file)

    def update_cell_plot(self, row_ids, selected_row_ids, data):
        #here we are are overriding the update_cell_plot function to add in the dandi data, allowing streaming from the dandi api
        selected_id_set = set(selected_row_ids or [])

        if row_ids is None:
            dff = self.df
            # pandas Series works enough like a list for this to be OK
            row_ids = self.df['id']
        else:
            dff = self.df.loc[row_ids]

        active_row_ids = selected_row_ids
        if active_row_ids is None or len(active_row_ids) == 0:
            active_row_ids = [self.df.iloc[0]['id']]
        if active_row_ids is not None:
            #determine the amount of different cells to plot, then make up to 4 subplots
            len_active_row_ids = len(active_row_ids)
            if len_active_row_ids == 1:
                fig = make_subplots(rows=1, cols=1, subplot_titles=selected_row_ids)
                plot_coords = [(1,1)]
            elif len_active_row_ids == 2:
                fig = make_subplots(rows=2, cols=1, subplot_titles=selected_row_ids)
                plot_coords = [(1,1), (2,1)]
            elif len_active_row_ids >= 3:
                fig = make_subplots(rows=2, cols=2,subplot_titles=selected_row_ids[:4])
                plot_coords = [(1,1), (1,2), (2,1), (2,2)]
            #now iter through the active row ids and plot them
            for active_row_id in active_row_ids[:4]:
                x, y,c = self.load_data(active_row_id)
                
                traces = []
                for sweep_x, sweep_y in zip(x, y):
                    traces.append(go.Scatter(x=sweep_x, y=sweep_y, mode='lines', hoverinfo='skip'))
                fig.add_traces(traces, rows=plot_coords[active_row_ids.index(active_row_id)][0], cols=plot_coords[active_row_ids.index(active_row_id)][1])
                fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
            fig.update_yaxes(automargin=True)
            fig.layout.autosize = True
            return dcc.Graph(
                id="file_plot",
                figure=fig,
                style={
                    "width": "100%",
                    "height": "100%"
                },
                config=dict(
                    autosizable=True,
                    frameMargins=0,
                ),
                responsive=True
            )

    def load_data(self, specimen_id):
        #here we are overriding  in the dandi data, allowing streaming from the dandi api
        if specimen_id is None:
            return self.df
        
        # first, create a virtual filesystem based on the http protocol and use
        # we need to parse the speciemen id to get the 
        dandiset_id = specimen_id.split('//')[1].split('/')[0]
        filepath = '/'.join(specimen_id.split('//')[1].split('/')[1:])
        with DandiAPIClient() as client:
            asset = client.get_dandiset(dandiset_id, 'draft').get_asset_by_path(filepath)
            s3_url = asset.get_content_url(follow_redirects=1, strip_query=True)
        # next, open the file
        with fs.open(s3_url, "rb") as f:
            _, _, _, _, data_set = loadNWB(f, return_obj=True)
            sweeps, start_times, end_times = parse_long_pulse_from_dataset(data_set)
            start_time = scipy.stats.mode(np.array(start_times))[0][0]
            end_time = scipy.stats.mode(np.array(end_times))[0][0]
            idx_pass = np.where((np.array(start_times) == start_time) & (np.array(end_times) == end_time))[0]
            #index out the sweeps that have the most common start and end times
            #take 10% of the stim epochs
            start_time *= 0.75
            end_time *= 1.2
            x = np.array([sweep.t[int(sweep.sampling_rate*start_time):int(sweep.sampling_rate*end_time)] for i, sweep in enumerate(sweeps) if i in idx_pass])
            y = np.array([sweep.v[int(sweep.sampling_rate*start_time):int(sweep.sampling_rate*end_time)] for i, sweep in enumerate(sweeps) if i in idx_pass])
            c = np.array([sweep.i[int(sweep.sampling_rate*start_time):int(sweep.sampling_rate*end_time)] for i, sweep in enumerate(sweeps) if i in idx_pass])
        if len(y) > 10:
            #grab every n sweeps so the length is about 10
            idx = np.arange(0, len(y), int(len(y)/10))
            x = x[idx]
            y = y[idx]
            c = c[idx]
        return x, y, c
    
    def _generate_header(self):
        return dbc.Row([
            dbc.Col(html.H1("Dandi icephys", className="text-center"), width=12),
            dbc.Col(html.H3("Live Data Visualization",
                    className="text-center"), width=12),
            dbc.Col(html.H5("Select a file to view",
                    className="text-center"), width=12),
        ])


#run_merge_dandiset()
    # caching to save accessed data to RAM.
fs = CachingFileSystem(
    fs=fsspec.filesystem("http"),
    cache_storage="nwb-cache",  # Local folder for the cache
)

GLOBAL_STIM_NAMES.stim_inc =['']

GLOBAL_VARS.file_index = 'specimen_id'
GLOBAL_VARS.file_path = 'specimen_id'
GLOBAL_VARS.table_vars_rq = ['specimen_id', 'dandiset label', 'species', 'GMM cluster label']
GLOBAL_VARS.table_vars = ['ap_1', 'resist']
GLOBAL_VARS.para_vars = ['ap_1', 'resist']
GLOBAL_VARS.para_var_colors = 'ap_1_width_0_long_square'
GLOBAL_VARS.umap_labels = ['dandiset label', 'species', 'GMM cluster label']


dandi_data_viz2 = dandi_data_viz(database_file='/media/smestern/Expansion/dandi/all.csv')
server = dandi_data_viz2.app.server

    