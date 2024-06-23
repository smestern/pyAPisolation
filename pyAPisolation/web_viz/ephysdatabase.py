import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import bs4
import json
import sys
from scipy.signal import resample, decimate
from pyAPisolation.patch_ml import *
from pyAPisolation.patch_utils import *
from pyAPisolation.feature_extractor import *
from pyAPisolation.loadFile import loadABF
import pyabf
from http.server import HTTPServer, CGIHTTPRequestHandler
import matplotlib.pyplot as plt
import anndata as ad
from .web_viz_config import web_viz_config
from .tsdatabase import tsDatabase
import shutil
from .flask_app import tsServer
import logging
logger = logging.getLogger(__name__)

_LOCAL_PATH = os.path.dirname(os.path.abspath(__file__))

def gen_table_head_str_(col, soup, dict_args=None):
    tag = soup.new_tag(f"th")
    tag['data-field'] = f"{col}"
    tag['data-sortable'] = f"true"
    tag['searchable'] = f"true"

    if dict_args is not None:
        for key, value in dict_args.items():
            tag[key] = value
    tag.string = f"{col}"
    return tag 

def generate_plots(df, static, filename='filename', foldername='foldername.1'):
    ids = df['filename'].to_numpy()
    folders = df['foldername.1'].to_numpy()
    full_y = []   
    for f, fp in zip(ids, folders):
        x, y, z = loadABF(os.path.join(fp,f+'.abf')) 
        y = decimate(y, 4, axis=1)
        x = decimate(x, 4, axis=1)
        idx = np.argmin(np.abs(x-2.5))
        y = y[:, :idx]
        y = np.vstack((x[0, :idx], y))
        fp = create_dir(f"./data/")
        fp += f"{f}.csv"
        if not static:
            np.savetxt(fp, y, delimiter=',', fmt='%.8f')
        elif static:
            
            full_y.append(fp)
    return full_y


class ephysDatabase (tsDatabase):
    def __init__(self, database_file, config=None, **kwargs):
        self.super().__init__(database_file, config, **kwargs)


def main(database_file=None, config=None, static=False):
    """Main function to run the web visualization. This script can be run from the command line or imported as a module.
    The function will generate a web page with the data from the database file. The data will be displayed in a table. Page can be exported static for use with github pages etc.
    takes:
        database_file: str, path to the database file
        config: dict, configuration for the web visualization
        static: bool, if True, the plots will be generated and saved in a separate folder, and then loaded into the html file. If False, the plots will be served via flask
    returns:
        None
    """
    #if the user does not provide a database file, ask for one
    if database_file is None:
        files = filedialog.askopenfilenames(filetypes=(('All files', '*.*'), ('Csv Files', '*.csv'),),
                                            title='Select Input File')
    else: 
        files = [database_file]
    if config is None:
        config = web_viz_config()
    elif isinstance(config, str):
        config = web_viz_config(file=config)
    elif isinstance(config, dict):
        config = web_viz_config(**config)
    
    files = list(files)
    fileList = files

    #load the data
    full_dataframe = pd.DataFrame()
    for x in fileList:
        if x.endswith('.csv'):
            temp_df = pd.read_csv(x)
        elif x.endswith('.xlsx'):
            temp_df = pd.read_excel(x)
        elif x.endswith('.h5ad'):
            temp = ad.read_h5ad(x)
            temp_df = temp.to_df()
            temp_df = temp_df.join(temp.obs)
        else:
            print("File type not supported")
            return
        full_dataframe = full_dataframe.append(temp_df)

    ### Preprocess the data, 
    #get the columns that are required by the config    
    full_dataframe = df_select_by_col(full_dataframe, [*config.table_vars_rq, *config.table_vars, *config.umap_labels, *config.para_vars])
    full_dataframe['ID'] = full_dataframe[config.file_index] if config.file_index in full_dataframe.columns else full_dataframe.index #add an ID column, if it does not exist
    bool_l = [x=='label' for x in full_dataframe.columns.values]
    if np.any(bool_l):
        labels = full_dataframe['label'].to_numpy()
    else:
        labels = None
    pred_col, labels = extract_features(full_dataframe.select_dtypes(["float32", "float64", "int32", "int64"]), ret_labels=True, labels=labels)
    full_dataframe['label'] = labels

    #check if the umap_cols are present in the dataframe
    if not all([x in full_dataframe.columns for x in config.umap_cols]):
        print("Umap columns not present in the dataframe, generating...")
        #make the umap columns
        data, outlier_idx = preprocess_df(full_dataframe.select_dtypes(["float32", "float64", "int32", "int64"]))
        umap_data = dense_umap(data)
        #insert nan values for the outliers
        for idx in outlier_idx:
            umap_data = np.insert(umap_data, idx, np.nan, axis=0)
        full_dataframe['Umap X'] = umap_data[:, 0]
        full_dataframe['Umap Y'] = umap_data[:, 1]
        print("Umap columns generated")
    else:
        umap_data = full_dataframe[config.umap_cols].to_numpy()

    ## handle plots
    if config.plots_path: #if the user has already pregeneated the plots
        plot_data = [os.path.join(config.plots_path, x+".csv") for x in full_dataframe['ID'].to_numpy()]
    else:
        plot_data = generate_plots(full_dataframe, static=static, filename=config.file_index, foldername=config.file_path)
    
    #Fix foldernames by truncating
    new_names = []
    for name in full_dataframe[config.file_path].to_numpy():
        temp = name#os.path.split(name)[0]
        new_names.append(temp)
    full_dataframe['foldername'] = new_names #add the new foldername column

    #convert the dataframe to json
    json_df = full_dataframe.to_json(orient='records')
    parsed = json.loads(json_df)
    if static:
        for i, dict_data in enumerate(parsed):
            dict_data['y'] = plot_data[i]
            for key, value in dict_data.items():
                if isinstance(value, np.int64):
                    dict_data[key] = int(value)
                elif isinstance(value, str):
                    dict_data[key] = value
                elif np.isscalar(value):
                    if value < 1 and value > -1:
                        dict_data[key] = round(value, 4)
                    else:
                        dict_data[key] = round(value, 2)

            parsed[i] = dict_data
    json_str = json.dumps(parsed)

    #open our template, and insert the json data
    TEMPLATE = os.path.join(_LOCAL_PATH, "template.html")# if not static else os.path.join(_LOCAL_PATH, "template_static.html")
    with open(TEMPLATE) as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt, 'html.parser')

    json_var = '  var data_tb = ' + json_str + ' '

    tag = soup.new_tag("script")
    tag.append(json_var)

    head = soup.find('body')
    head.insert_before(tag)

    #column tags
    table_head= soup.find('th')
    pred_col = np.hstack((pred_col[:10], 'foldername', *config.table_vars_rq, *config.table_vars))
    print(pred_col)
    for col in pred_col:
        logger.info(f"Adding column {col}")
        test = gen_table_head_str_(col, soup)
        table_head.insert_after(test)

    if not static:
        #replace the template.js import in the html with 
        #template_dyn.js import
        script_tag = soup.find_all('head')[0]
        #add a new script tag onto the end of the list
        new_tag = soup.new_tag('script')
        new_tag['src'] = 'assets/template_dyn.js'
        script_tag.append(new_tag)


    #export everything to the out_path_folder
    if not os.path.exists(config.output_path):
        os.makedirs(config.output_path)
    
    with open(os.path.join(config.output_path, "output.html"), "w") as outf:
        outf.write(str(soup))

    #copy the 'assets' folder
    shutil.copytree(os.path.join(_LOCAL_PATH, "assets"), os.path.join(config.output_path,"assets"), dirs_exist_ok=True)
        
    if static:
        print("=== Running Server ===")
        #Create server object listening the port 80
        server_object = HTTPServer(server_address=('', 80), RequestHandlerClass=CGIHTTPRequestHandler)
        #spawn a new thread for the server to run on
        server_object.server_activate()
        
        
        #start the server
        server_object.serve_forever()
    else:
        print("=== Running Server ===")
        tsServer(config=config, static=False).run()





if __name__ == '__main__':
    main()