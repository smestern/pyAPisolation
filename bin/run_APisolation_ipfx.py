
print("Loading...")
import sys


import logging
logging.getLogger().setLevel(logging.DEBUG)
import os
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyabf

from numpy import genfromtxt
print("Loaded external libraries")
from pyAPisolation.abf_featureextractor import folder_feature_extract, save_data_frames
from pyAPisolation.patch_utils import load_protocols


print("Load finished")





def main():
    logging.basicConfig(level=logging.DEBUG)
    root = tk.Tk()
    root.withdraw()

    files = filedialog.askdirectory(
                                    title='Select Dir'
                                    )
    root_fold = files

    ##Declare our options at default


    print('loading protocols...')
    protocol_n = load_protocols(files)
    print("protocols")
    for i, x in enumerate(protocol_n):
        print(str(i) + '. '+ str(x))
    proto = input("enter Protocol to analyze (enter -1 to not filter to any protocol): ")
    try: 
        proto = int(proto)
    except:
        proto = -1



    filter = input("Allen's Gaussian Filter (recommended to be set to 0): ")
    try: 
        filter = int(filter)
    except:
        filter = 0

    savfilter = input("Bessel Filter (recommended to be set in 5000): ")
    try: 
        savfilter = int(savfilter)
    except:
        savfilter = 0

    tag = input("tag to apply output to files: ")
    try: 
        tag = str(tag)
    except:
        tag = ""
    plot_sweeps = input("Enter the sweep Numbers to plot [seperated by a comma] (0 to plot all sweeps, -1 to plot no sweeps): ")
    try:
        plot_sweeps = np.fromstring(plot_sweeps, dtype=int, sep=',')
        if plot_sweeps.shape[0] < 1:
            plot_sweeps = np.array([-1])
    except:
        plot_sweeps = -1

    
    if proto == -1:
        protocol_name = ''
    else:
        protocol_name = protocol_n[proto]
    dv_cut = input("Enter the threshold cut off for the derivative (defaults to 7mv/s): ")
    try: 
        dv_cut = int(dv_cut)
    except:
        dv_cut = 7
    tp_cut = input("Enter the threshold cut off for max threshold-to-peak time (defaults to 10ms)[in ms]: ")
    try: 
        tp_cut = (np.float64(tp_cut)/1000)
    except:
        tp_cut = 0.010

    min_cut = input("Enter the minimum cut off for threshold-to-peak voltage (defaults to 2mV)[in mV]: ")
    try: 
        min_cut = np.float64(min_cut)
    except:
        min_cut = 2


    min_peak = input("Enter the mininum cut off for peak voltage (defaults to -10)[in mV]: ")
    try: 
        min_peak = np.float64(min_peak)
    except:
        min_peak = -10

    percent = input("Enter the percent of max DvDt used to calculate refined threshold (does not effect spike detection)(Allen defaults 5%)[in %]: ")
    try: 
        percent = percent /100
    except:
        percent = 5/100
    stim_find = input("Search for spikes based on applied Stimulus? (y/n): ")
    

    try: 
        if stim_find == 'y' or stim_find =='Y':
            bstim_find = True
        else:
            bstim_find = False
    except:
        bstim_find = False


    if bstim_find:
        upperlim = 0
        lowerlim = 0
    else:
        lowerlim = input("Enter the time to start looking for spikes [in s] (enter 0 to start search at beginning): ")
        upperlim = input("Enter the time to stop looking for spikes [in s] (enter 0 to search the full sweep): ")

        try: 
            lowerlim = float(lowerlim)
            upperlim = float(upperlim)
        except:
            upperlim = 0
            lowerlim = 0



    print(f"Running analysis with, dVdt thresh: {dv_cut}mV/s, thresh to peak max: {tp_cut}s, thresh to peak min height: {min_cut}mV, and min peak voltage: {min_peak}mV")
    param_dict = {'filter': filter, 'dv_cutoff':dv_cut, 'start': lowerlim, 'end': upperlim, 'max_interval': tp_cut, 'min_height': min_cut, 'min_peak': min_peak, 'thresh_frac': percent, 
    'stim_find': bstim_find, 'bessel_filter': savfilter}
    df = folder_feature_extract(files, param_dict, plot_sweeps, protocol_name)
    print(f"Ran analysis with, dVdt thresh: {dv_cut}mV/s, thresh to peak max: {tp_cut}s, thresh to peak min height: {min_cut}mV, and min peak voltage: {min_peak}mV")
    save_data_frames(df[0], df[1], df[2], root_fold, tag)
    settings_col = ['dvdt Threshold', 'threshold to peak max time','threshold to peak min height', 'min peak voltage', 'allen filter', 'sav filter', 'protocol_name']
    setdata = [dv_cut, tp_cut, min_cut, min_peak, filter, savfilter, protocol_name]
    settings_df =  pd.DataFrame(data=[setdata], columns=settings_col, index=[0])
    settings_df.to_csv(root_fold + '/analysis_settings_' + tag + '.csv')

    print("==== SUCCESS ====")
    input('Press ENTER to exit')

if __name__ == "__main__":
    main()
