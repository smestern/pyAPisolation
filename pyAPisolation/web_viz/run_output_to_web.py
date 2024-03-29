import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import bs4
import json
import sys
from scipy.signal import resample, decimate
sys.path.append('..')
sys.path.append('')
os.chdir("./pyAPisolation/")
print(os.getcwd())
from pyAPisolation.patch_ml import *
from pyAPisolation.patch_utils import *
from pyAPisolation.feature_extractor import *
from pyAPisolation.loadABF import loadABF
import pyabf
os.chdir("./web_viz")
from http.server import HTTPServer, CGIHTTPRequestHandler
import matplotlib.pyplot as plt
import anndata as ad



FILENAME_COL = "filename"
FOLDERNAME_COL = "foldername.1"
LABEL_COL = "label"


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

def generate_plots(df, static):
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


def main(database_file=None, static=False):
    if database_file is None:
        files = filedialog.askopenfilenames(filetypes=(('All files', '*.*'), ('Csv Files', '*.csv'),),
                                            title='Select Input File')
    else: 
        files = [database_file]
    files = list(files)
    fileList = files

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
    full_dataframe = df_select_by_col(full_dataframe, ['rheo', 'latency', 'filename', 'foldername'])



    full_dataframe['ID'] = full_dataframe['filename'] if 'filename' in full_dataframe.columns else full_dataframe.index 
    bool_l = [x=='label' for x in full_dataframe.columns.values]
    if np.any(bool_l):
        labels = full_dataframe['label'].to_numpy()
    else:
        labels = None

    pred_col, labels = extract_features(full_dataframe.select_dtypes(["float32", "float64", "int32", "int64"]), ret_labels=True, labels=labels)
    plot_data = generate_plots(full_dataframe, static=static)
    full_dataframe['label'] = labels
    #Fix foldernames by truncating
    new_names = []
    for name in full_dataframe['foldername.1'].to_numpy():
        temp = name.split("/")[-3:]
        #temp = temp[0] + "_" + temp[1] + "_" + temp[2]
        new_names.append(temp)
    full_dataframe['foldername'] = new_names

    json_df = full_dataframe.to_json(orient='records')
    parsed = json.loads(json_df)
    if static:
        for i, dict_data in enumerate(parsed):
            dict_data['y'] = plot_data[i]
            parsed[i] = dict_data
    json_str = json.dumps(parsed)  
    TEMPLATE = "template.html" if not static else "template_static.html"
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
    pred_col = np.hstack((pred_col[:10], 'foldername'))
    print(pred_col)
    for col in pred_col:
        test = gen_table_head_str_(col, soup)
        table_head.insert_after(test)

    with open("output.html", "w") as outf:
        outf.write(str(soup))
        
    print("=== Running Server ===")
    #Create server object listening the port 80
    server_object = HTTPServer(server_address=('', 80), RequestHandlerClass=CGIHTTPRequestHandler)
    #Start the web server
    server_object.serve_forever()


if __name__ == '__main__':
    main()