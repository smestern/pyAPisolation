



import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import bs4
import json
import sys
from scipy.signal import decimate
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

index_col = "filename"

def loadABF(file_path, return_obj=False):
    '''
    Employs pyABF to generate numpy arrays of the ABF data. Optionally returns abf object.
    Same I/O as loadNWB
    '''
    abf = pyabf.ABF(file_path)
    dataX = []
    dataY = []
    dataC = []
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        tempX = abf.sweepX
        tempY = abf.sweepY
        tempC = abf.sweepC
        dataX.append(tempX)
        dataY.append(tempY)
        dataC.append(tempC)
    npdataX = np.vstack(dataX)
    npdataY = np.vstack(dataY)
    npdataC = np.vstack(dataC)

    if return_obj == True:

        return npdataX, npdataY, npdataC, abf
    else:

        return npdataX, npdataY, npdataC

    ##Final return incase if statement fails somehow
    return npdataX, npdataY, npdataC


def gen_table_head_str_(col, soup):
    tag = soup.new_tag(f"th")
    tag['data-field'] = f"{col}"
    tag['data-sortable'] = f"true"
    tag.string = f"{col}"
    return tag #f"<th data-field=\"{col}\">{col}</th> "

def generate_plots(df):
    ids = df['filename'].to_numpy()
    folders = df['foldername'].to_numpy()
    full_y = []
    for f, fp in zip(ids, folders):
        x, y, z = loadABF(os.path.join(fp,f+'.abf'))
        y = decimate(y, 4, axis=1)
        x = decimate(x, 4, axis=1)
        idx = np.argmin(np.abs(x-2.5))
        #y = np.round(y, 1)
        y = y[:, :idx]
        y = np.vstack((x[0, :idx], y))
        y = y.tolist()
        y = [[round(x, 1) for x in l] for l in y]
        full_y.append(y)
    return full_y


def main():
    files = filedialog.askopenfilenames(filetypes=(('ABF Files', '*.csv'),
                                    ('All files', '*.*')),
                                    title='Select Input File'
                                    )
    fileList = files

    full_dataframe = pd.DataFrame()
    for x in fileList:
        temp = pd.read_csv(x, )
        full_dataframe = full_dataframe.append(temp)
    #full_dataframe = full_dataframe.set_index(index_col)
    #full_dataframe = full_dataframe.select_dtypes(["float32", "float64", "int32", "int64"])
    #full_dataframe = full_dataframe.drop(labels=['Unnamed: 0'], axis=1)
    full_dataframe['ID'] = full_dataframe[index_col]
    pred_col = extract_features(full_dataframe.select_dtypes(["float32", "float64", "int32", "int64"]), ret_labels=True)
    
    plot_data = generate_plots(full_dataframe)
    json_df = full_dataframe.to_json(orient='records')
    parsed = json.loads(json_df)
    for i, dict_data in enumerate(parsed):
        dict_data['y'] = plot_data[i]
        parsed[i] = dict_data
    json_str = json.dumps(parsed)  
    with open("template_static.html") as inf:
        txt = inf.read()
        soup = bs4.BeautifulSoup(txt, 'html.parser')

    json_var = '  var data_tb = ' + json_str + ' '

    tag = soup.new_tag("script")
    tag.append(json_var)

    head = soup.find('body')


    head.insert_before(tag)

    #column tags
    table_head= soup.find('th')
    
    for col in pred_col[:5]:
        test = gen_table_head_str_(col, soup)
        table_head.insert_after(test)

    with open("output.html", "w") as outf:
        outf.write(str(soup))
main()