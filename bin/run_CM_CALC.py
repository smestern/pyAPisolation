
print("Loading...")
import sys
import numpy as np
from numpy import genfromtxt
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from multiprocessing import Process, freeze_support
from ipfx import subthresh_features as subt
import pyabf
import logging
from pyAPisolation.patch_subthres import *
from pyAPisolation.QC import *
from pyAPisolation.feature_extractor import _merge_current_injection_features
print("Load finished")
def main():
    
    logging.basicConfig(level=logging.DEBUG)
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askdirectory(
                                    title='Select dir File'
                                    )
    root_fold = files

    print('loading protocols...')
    protocol = []
    for root,dir,fileList in os.walk(files):
        for filename in fileList:
            if filename.endswith(".abf"):
                try:
                    file_path = os.path.join(root,filename)
                    abf = pyabf.ABF(file_path, loadData=False)
                    protocol = np.hstack((protocol, abf.protocol))
                except:
                    print('error processing file ' + file_path)
    protocol_n = np.unique(protocol)
    filter = input("Allen's Gaussian Filter (recommended to be set to 0): ")
    braw = False
    bfeat = True
    try: 
        filter = int(filter)
    except:
        filter = 0

    savfilter = input("Savitzky-Golay Filter (recommended to be set in 0): ")
    braw = False
    bfeat = True
    try: 
        savfilter = int(savfilter)
    except:
        savfilter = 0

    _sbplot = input("Plot the fit(s) (y/n): ")

    try: 
        if _sbplot == 'y' or _sbplot =='Y':
            bplot = True
        else:
            bplot = False
    except:
        bplot = False

    tag = input("tag to apply output to files: ")
    try: 
        tag = str(tag)
    except:
        tag = ""

    plot_sweeps = -1

    print("protocols")
    for i, x in enumerate(protocol_n):
        print(str(i) + '. '+ str(x))
    proto = input("enter Protocol to analyze (enter -1 to not filter down to any protocol): ")
    try: 
        proto = int(proto)
    except:
        proto = -1


    protocol_name = protocol_n[proto]

    time_after = input("Enter the percentage of the stim time to include (default 50%): ")
    try: 
        time_after = int(time_after)
    except:
        time_after = 50

    subt_sweeps = input("Enter subthreshold sweeps (seperated by comma), if None program will try to guess: ")
    try: 
        subt_sweeps = np.fromstring(subt_sweeps, dtype=int, sep=',')
        if subt_sweeps.shape[0] < 1:
            subt_sweeps = None
    except:
        subt_sweeps = None
            
        time_after = 50

    start_sear = input("Enter the time to begin analyzing in the protocol: ")
    try: 
        start_sear = float(start_sear)
    except:
        start_sear = None

    end_sear = input("Enter the time to stop analyzing in the protocol: ")
    try: 
        end_sear = float(end_sear)
    except:
        end_sear = None    

    if bplot == True:
                            if not os.path.exists(root_fold+'//cm_plots//'):
                                    os.mkdir(root_fold+'//cm_plots//')  

    
    dfs = pd.DataFrame()
    averages = pd.DataFrame()
    for root,dir,fileList in os.walk(files):
        for filename in fileList:
            if filename.endswith(".abf"):
                file_path = os.path.join(root,filename)
                if True: #try:
                    abf = pyabf.ABF(file_path)
                    plt.close('all')
                    if (proto==-1) or (abf.sweepLabelY != 'Clamp Current (pA)'and abf.protocol != 'Gap free' and protocol_name in abf.protocol):
                        print(filename + ' import')
                    
                        np.nan_to_num(abf.data, nan=-9999, copy=False)
                        if savfilter >0:
                            abf.data = signal.savgol_filter(abf.data, savfilter, polyorder=3)
                        
                        #determine the search area
                        dataT = abf.sweepX #sweeps will need to be the same length
                        if start_sear != None:
                            idx_start = np.argmin(np.abs(dataT - start_sear))
                        else:
                            idx_start = 0
                            
                        if end_sear != None:
                            idx_end = np.argmin(np.abs(dataT - end_sear))
                            
                        else:
                            idx_end = -1



                        #If there is more than one sweep, we need to ensure we dont iterate out of range
                        if abf.sweepCount > 1:
                            if subt_sweeps is None:
                                sweepList = determine_subt(abf, (idx_start, idx_end))
                                sweepcount = len(sweepList)
                            else:
                                subt_sweeps_temp = subt_sweeps - 1
                                sweep_union = np.intersect1d(abf.sweepList, subt_sweeps_temp)
                                sweepList = sweep_union
                                sweepcount = 1
                        else:
                            sweepcount = 1
                            sweepList = [0]
                        
                        #Now we walk through the sweeps looking for action potentials
                        temp_df = pd.DataFrame()
                        temp_df['1Afilename'] = [abf.abfID]
                        temp_df['1Afoldername'] = [os.path.dirname(file_path)]
                        temp_avg = pd.DataFrame()
                        temp_avg['1Afilename'] = [abf.abfID]
                        temp_avg['1Afoldername'] = [os.path.dirname(file_path)]
                        
                        full_dataI = []
                        full_dataV = []
                        for sweepNumber in sweepList: 
                            real_sweep_length = abf.sweepLengthSec - 0.0001
                            if sweepNumber < 9:
                                real_sweep_number = '00' + str(sweepNumber + 1)
                            elif sweepNumber > 8 and sweepNumber < 99:
                                real_sweep_number = '0' + str(sweepNumber + 1)

                            

                            abf.setSweep(sweepNumber)
                    
                            dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
                            dataT, dataV, dataI = dataT[idx_start:idx_end], dataV[idx_start:idx_end], dataI[idx_start:idx_end]
                            dataT = dataT - dataT[0]
                            

                            decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, p_decay = exp_decay_factor(dataT, dataV, dataI, time_after, abf_id=abf.abfID)
                            
                            resist = membrane_resistance(dataT, dataV, dataI)
                            Cm2, Cm1 = mem_cap(resist, decay_slow)
                            Cm3 = mem_cap_alt(resist, decay_slow, curve[3], np.amin(dataI))
                            temp_df[f"_1 phase tau {real_sweep_number}"] = [p_decay]           
                            temp_df[f"fast 2 phase tau {real_sweep_number}"] = [decay_fast]
                            temp_df[f"slow 2 phase tau {real_sweep_number}"] = [decay_slow]
                            temp_df[f"Curve fit A {real_sweep_number}"] = [curve[0]]
                            temp_df[f"Curve fit b1 {real_sweep_number}"] = [curve[1]]
                            temp_df[f"Curve fit b2 {real_sweep_number}"] = [curve[3]]
                            temp_df[f"R squared 2 phase {real_sweep_number}"] = [r_squared_2p]
                            temp_df[f"R squared 1 phase {real_sweep_number}"] = [r_squared_1p]
                            temp_df[f"RMP {real_sweep_number}"] = [rmp_mode(dataV, dataI)]
                            temp_df[f"Membrane Resist {real_sweep_number}"] =  resist / 1000000000 #to gigaohms
                            temp_df[f"_2 phase Cm {real_sweep_number}"] =  Cm2 * 1000000000000#to pf farad
                            temp_df[f"_ALT_2 phase Cm {real_sweep_number}"] =  Cm3 * 1000000000000
                            temp_df[f"_1 phase Cm {real_sweep_number}"] =  Cm1 * 1000000000000
                            temp_df[f"Voltage sag {real_sweep_number}"],temp_df[f"Voltage min {real_sweep_number}"] = compute_sag(dataT,dataV,dataI, time_after, plot=bplot, clear=False)
                            try:
                                sag_ratio, taum_allen, voltage_allen = subthres_a(dataT,dataV,dataI, 0.0, np.amax(dataT))
                                temp_df[f"Voltage sag ratio {real_sweep_number}"] = sag_ratio
                                temp_df[f"Tau_m Allen {real_sweep_number}"] = taum_allen    
                                temp_df[f"Voltage sag Allen {real_sweep_number}"] = voltage_allen[0]
                            except:
                                temp_df[f"Voltage sag ratio {real_sweep_number}"] = np.nan
                                temp_df[f"Tau_m Allen {real_sweep_number}"] = np.nan
                                temp_df[f"Voltage sag Allen {real_sweep_number}"] = np.nan
                                    #temp_spike_df['baseline voltage' + real_sweep_number] = subt.baseline_voltage(dataT, dataV, start=b_lowerlim)
                            #
                            #temp_spike_df['time_constant' + real_sweep_number] = subt.time_constant(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                            #temp_spike_df['voltage_deflection' + real_sweep_number] = subt.voltage_deflection(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)



                            full_dataI.append(dataI)
                            full_dataV.append(dataV)
                            if dataI.shape[0] < dataV.shape[0]:
                                dataI = np.hstack((dataI, np.full(dataV.shape[0] - dataI.shape[0], 0)))
                            sweepcount
                        if bplot == True:
                                plt.title(abf.abfID)
                                plt.ylim(top=-40)
                                plt.xlim(right=0.6)
                                #plt.legend()
                                plt.savefig(root_fold+'//cm_plots//sagfit'+abf.abfID+'sweep'+real_sweep_number+'.png')
                        
                        full_dataI = np.vstack(full_dataI) 
                        #indices_of_same = np.any(full_dataI, axis=1)
                        #if np.any(indices_of_same) == False:
                        #    indices_of_same = 0
                        indices_of_same = np.arange(full_dataI.shape[0])
                        full_dataV = np.vstack(full_dataV)
                         
                        print("Fitting tau")
                        decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, p_decay = exp_decay_factor_alt(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), 
                        np.nanmean(full_dataI[indices_of_same,:],axis=0), time_after, abf_id=abf.abfID, plot=bplot, root_fold=root_fold)
                        print("Computing Sag")
                        #grow = exp_growth_factor(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0), 1/decay_slow)
                        temp_avg[f"Voltage sag mean"], temp_avg["Voltage Min point"] = compute_sag(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0), time_after, plot=bplot)
                        temp_avg[f"Sweepwise Voltage sag mean"], temp_avg["Sweepwise Voltage Min point"] = np.nanmean(df_select_by_col(temp_df, ['Voltage sag 0'])), np.nanmean(df_select_by_col(temp_df, ['Voltage min']))
                        if bplot == True:
                            plt.title(abf.abfID)
                            plt.savefig(root_fold+'//cm_plots//sagfit'+abf.abfID)
                        temp_avg["Averaged 1 phase tau "] = [p_decay]           
                        temp_avg["Averaged 2 phase fast tau "] = [decay_fast]
                        temp_avg["Averaged 2 phase slow tau "] = [decay_slow]
                        temp_avg["Averaged Curve fit A"] = [curve[0]]
                        temp_avg["Averaged Curve fit b1"] = [curve[1]]
                        temp_avg["Averaged Curve fit b2"] = [curve[3]]
                        temp_avg["Averaged R squared 2 phase"] = [r_squared_2p]
                        temp_avg["Averaged R squared 1 phase"] = [r_squared_1p]
                        temp_avg[f"Averaged RMP"] = [rmp_mode(np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0))]
                        temp_avg["SweepCount Measured"] = [sweepcount]
                        #temp_avg["Averaged alpha tau"] = [grow[1]]
                        #temp_avg["Averaged b tau"] = [grow[3]]
                        if r_squared_2p > r_squared_1p:
                            temp_avg["Averaged Best Fit"] = [2]
                        else:
                            temp_avg["AverageD Best Fit"] = [1]
                        print(f"fitting Membrane resist")
                        resist = membrane_resistance(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0))
                        resist_alt = exp_rm_factor(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0), time_after, decay_slow, abf_id=abf.abfID,  root_fold=root_fold)
                        Cm2, Cm1 = mem_cap(resist, decay_slow, p_decay)
                        Cm3 = mem_cap_alt(resist, decay_slow, curve[3], np.amin(np.nanmean(full_dataI[indices_of_same,:],axis=0)))
                        rm_alt = mem_resist_alt(Cm3, decay_slow)
                        temp_avg["Averaged Membrane Resist"] =  resist  / 1000000000 #to gigaohms
                        temp_avg["Averaged Membrane Resist _ ALT"] =  resist_alt[0]  / 1000000000
                        temp_avg["Averaged Membrane Resist _ ALT 2"] =  rm_alt  / 1000000000#to gigaohms
                        temp_avg["Averaged pipette Resist _ ALT"] =  resist_alt[2]  / 1000000000 #to gigaohms
                        temp_avg["Averaged 2 phase Cm"] =  Cm2 * 1000000000000
                        temp_avg["Averaged 2 phase Cm Alt"] =  Cm3 * 1000000000000
                        temp_avg["Averaged 1 phase Cm"] =  Cm1 * 1000000000000
                        try:
                            sag_ratio, taum_allen, voltage_allen = subthres_a(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0),
                                                                    np.nanmean(full_dataI[indices_of_same,:],axis=0), 0.0, np.amax(dataT))
                            temp_avg[f"Averaged Voltage sag ratio "] = sag_ratio
                            temp_avg[f"Averaged Tau_m Allen "] = taum_allen    
                            temp_avg[f"Average Voltage sag Allen "] = voltage_allen[0]
                        except:
                            temp_avg[f"Averaged Voltage sag ratio "] = np.nan
                            temp_avg[f"Averaged Tau_m Allen "] = np.nan
                            temp_avg[f"Average Voltage sag Allen "] = np.nan

                        #compute the QC features
                        print("Computing QC features")
                        try:
                            mean_rms, max_rms, mean_drift, max_drift = run_qc(full_dataV[indices_of_same,:], full_dataI[indices_of_same,:])
                            temp_avg["Averaged Mean RMS"] = mean_rms
                            temp_avg["Max RMS"] = max_rms
                            temp_avg["Averaged Mean Drift"] = mean_drift
                            temp_avg["Max Drift"] = max_drift
                        except:
                            temp_avg["Averaged Mean RMS"] = np.nan
                            temp_avg["Max RMS"] = np.nan
                            #temp_avg["Averaged Mean Drift"] = np.nan
                            #temp_avg["Max Drift"] = np.nan
                    #pack in some protocol info
                        temp_avg = _merge_current_injection_features(sweepX=np.tile(dataT, (full_dataI.shape[0], 1)), sweepY=full_dataI, sweepC=full_dataI, spike_df=temp_avg)
                        

                        #try the ladder_RM
                        print("Computing ladder RM")
                        try:
                            rm_ladder, _, sweep_count = ladder_rm(np.tile(dataT, (full_dataI.shape[0], 1)), full_dataV, full_dataI)
                            temp_avg["Resistance Ladder Slope"] = rm_ladder
                            temp_avg["Rm Resistance Ladder"] = 1/rm_ladder
                            temp_avg["Resistance Ladder SweepCount Measured"] = sweep_count
                        except:
                            temp_avg["Resistance Ladder Slope"] = np.nan
                            temp_avg["Rm Resistance Ladder"] = np.nan
                            temp_avg["Resistance Ladder SweepCount Measured"] = np.nan
                        
                        print(f"Computed a membrane resistance of {(resist  / 1000000000)} giga ohms, and a capatiance of {Cm2 * 1000000000000} pF, and tau of {decay_slow*1000} ms")
                        dfs = dfs.append(temp_df, sort=True)
                        averages = averages.append(temp_avg, sort=True)
                    #except:
                    # print('Issue Processing ' + filename)

                    else:
                        print('Not correct protocol: ' + abf.protocol)
                #except:
                   # print('Issue Processing ' + filename)


    if True:
        #try:
        dfs = dfs.reindex(sorted(dfs.columns), axis=1)
        averages = averages.reindex(sorted(averages.columns), axis=1)
        #dfs.to_csv(root_fold + f'/Membrane_cap_{tag}.csv')
        with pd.ExcelWriter(root_fold + '/mem_cap_' + tag + '.xlsx') as runf:
            averages.set_index('1Afilename').to_excel(runf, sheet_name='Averages')
            dfs.set_index('1Afilename').to_excel(runf, sheet_name='Sweepwise Calculations')
        



    #except:
    # print('error saving')

    print("==== SUCCESS ====")
    input('Press ENTER to exit')

if __name__=="__main__":
    freeze_support()
    
    main()