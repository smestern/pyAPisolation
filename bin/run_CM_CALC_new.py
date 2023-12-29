
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
import time
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import mode
from ipfx import subthresh_features as subt
from ipfx import feature_extractor
import pyabf
import logging
import scipy.ndimage as ndimage
from pyAPisolation.patch_subthres import subthres_a
print("Load finished")
logging.basicConfig(level=logging.DEBUG)
root = tk.Tk()
root.withdraw()
files = filedialog.askdirectory(
                                   title='Select Dir'
                                   )
root_fold = files

##Declare our options at default

def exp_grow(t, a, b, alpha):
    return a - (b * np.exp(-t/alpha)) 
def exp_grow_2p(t, a, b1, alphaFast, b2, alphaSlow):
    return a - b1*(np.exp(-t/alphaFast)) - b2*(np.exp(-t/alphaSlow))
def exp_decay_2p(t, a, b1, alphaFast, b2, alphaSlow):
    v = a + b1*(1-np.exp(-t/alphaFast)) + b2*(1-np.exp(-t/alphaSlow))
    return v * 1000
def exp_decay_1p(t, a, b1, alphaFast):
    return (a + b1*(1-np.exp(-t/alphaFast))) * 1000

def rm_decay_2p(t, Iinj, Rm, alphaFast, Re, alphaSlow, a):
    v = a + (Iinj * (Rm * (1 - np.exp(-t/alphaFast)) + Re * (1 - np.exp(-t/alphaSlow))))
    v = v * 1000#in Volts to mV
    return v

def exp_growth_factor(dataT,dataV,dataI, alpha, end_index=1, plot=False):
    try:
        dt = dataT[1] - dataT[0]
        end_index = int(end_index / dt)
        diff_I = np.diff(dataI)
        upwardinfl = np.argmax(diff_I)
        downwardinfl = np.argmin(diff_I[upwardinfl:])
        end_index = upwardinfl + int(downwardinfl * 0.95)
        #end_index += upwardinfl
        try:
            spikext = feature_extractor.SpikeFeatureExtractor(filter=0)
            spikes = spikext.process(dataT,dataV,dataI)
            first_spike_index = spikes['threshold_index'].to_numpy()[0]
            end_index = first_spike_index
        except:
            print("no spikes")
        upperC = np.amax(dataV[upwardinfl:end_index])
        lowerC = np.amin(dataV[upwardinfl:end_index])
        t1 = dataT[upwardinfl:end_index] - dataT[upwardinfl]
        _height = (upperC - lowerC) * 0.90
        curve2 = curve_fit(exp_grow, t1, dataV[upwardinfl:end_index], maxfev=50000, bounds=([-np.inf, _height - 5, 0], [np.inf, _height + 5, 0.1]), xtol=None)[0]
        curve = curve_fit(exp_grow_2p, t1, dataV[upwardinfl:end_index], maxfev=50000, bounds=([-np.inf, (_height - 5), 0, 0, 0], [0, _height*2, curve2[2], _height*2, 5]), xtol=None)[0]
        tau = curve[2]
        x_deriv, deriv_ar = deriv(t1, exp_grow(t1, *curve2))
        diff = np.abs(deriv_ar - 2)
        minpoint = np.argmin(diff)
        if plot==True:
            
            plt.figure(2)
            plt.clf()
            plt.plot(t1, dataV[upwardinfl:end_index], label='Data')
            plt.scatter(t1[minpoint], exp_grow(t1, *curve2)[minpoint], label='min')
            plt.plot(t1, exp_grow(t1, *curve2), label='1 phase fit')
            plt.plot(t1, exp_grow_2p(t1, *curve), label='2 phase fit')
            plt.legend()
            plt.title(abf_id)
            plt.pause(2)
            #plt.savefig(root_fold+ '//cm_plots//' + abf_id+'.png')
            #plt.close()
        tau1 = curve[2]
        tau2 = curve[4]
        tau_1p = curve2[2]
        fast = np.argmin([tau1, tau2])
        slow = np.argmax([tau1, tau2])
        _positions = [[1,2], [3,4]]
        fast_pos = _positions[fast]
        slow_pos = _positions[slow]
        curve_ordered = np.hstack((curve[0], curve[fast_pos], curve[slow_pos]))
        return np.hstack((curve_ordered, (t1[0]-t1[minpoint])))
    except:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


def deriv(x,y):
    dy=np.diff(y,1)
    dx=np.diff(x,1)
    yfirst=dy/dx
    xfirst=0.5*(x[:-1]+x[1:])
    return xfirst, yfirst

def rmp_mode(dataV, dataI):
    pre = find_downward(dataI)
    mode_vm = mode(dataV[:pre], nan_policy='omit')[0][0]
    return mode_vm

def mem_resist_alt(cm_alt, slow_decay):
    rm_alt = cm_alt / slow_decay
    return 1/rm_alt

def exp_rm_factor(dataT,dataV,dataI, time_aft, decay_slow, abf_id='abf', plot=False, root_fold=''):
    try:
        time_aft = time_aft / 100
        if time_aft > 1:
            time_aft = 1

        diff_I = np.diff(dataI)
        downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
        end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
        end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
        upperC = np.amax(dataV[downwardinfl:end_index])
        lowerC = np.amin(dataV[downwardinfl:end_index])
        minpoint = np.argmin(dataV[downwardinfl:end_index])
        end_index = downwardinfl + int(.95 * minpoint)
        downwardinfl = downwardinfl #+ int(.10 * minpoint)
        
        upperC = np.amax(dataV[downwardinfl:end_index])
        lowerC = np.amin(dataV[downwardinfl:end_index])
        diff = upperC/1000 + -1* np.abs(upperC - lowerC) /1000 + -0.005
        Iinj_real = dataI[int(downwardinfl + 5)] / 1000000000000 #in pA -> A
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        SpanFast=(upperC-lowerC)*1*.01
        if np.isnan(decay_slow):
            curve, pcov_2p = curve_fit(lambda t1, Rm, alphaFast, Re, alphaSlow, a: rm_decay_2p(t1, Iinj_real, Rm, alphaFast, Re, alphaSlow, a), t1, dataV[downwardinfl:end_index], 
                                    maxfev=50000, bounds=([1000000, 0, 0, 0, (upperC /1000)-0.002], [np.inf,  decay_slow,np.inf, decay_slow, (upperC /1000)+0.002]), verbose=1, xtol=None)
        else:
            guess = np.array([1000000000, decay_slow - 0.005, 5924871.10992442, np.abs(decay_slow-0.005), (upperC /1000)])
   
            curve, pcov_2p = curve_fit(lambda t1, Rm, alphaFast, Re, alphaSlow, a: rm_decay_2p(t1, Iinj_real, Rm, alphaFast, Re, alphaSlow, a), t1, dataV[downwardinfl:end_index], p0=guess,
                                    maxfev=50000, bounds=([1000000, 0, 0, 0, (upperC /1000)-0.002], [np.inf,  decay_slow,np.inf, decay_slow, (upperC /1000)+0.002]), verbose=1, xtol=None)

        #residuals_2p = dataV[downwardinfl:end_index]- exp_decay_2p(t1, *curve)
        
        if plot==True:

            plt.figure(42)
            plt.clf()
            plt.plot(t1, dataV[downwardinfl:end_index], label='Data')
            plt.plot(t1, rm_decay_2p(t1, Iinj_real, *curve), label='2 phase fit')
            #plt.plot(t1, exp_decay_1p(t1, curve[0], curve[3]/4, curve[2]) + np.abs(upperC - np.amax(exp_decay_1p(t1, curve[0], curve[3]/4, curve[2]))), label='Phase 1', zorder=9999)
            #plt.plot(t1, exp_decay_1p(t1, curve[0], curve[3], curve[4]) + np.abs(upperC - np.amax(exp_decay_1p(t1, curve[0], curve[3], curve[4]))), label='Phase 2')
            plt.legend()
            plt.title(abf_id + " RM decay")
            plt.pause(3)
            plt.savefig(root_fold+ '//cm_plots//' + abf_id+'_rm fit.png')
            plt.close() 
        max_point = np.argmax([curve[0], curve[2]])
        min_point = np.argmin([curve[0], curve[2]])
        _positions = [[0,1], [2,3]]
        fast_pos = _positions[max_point]
        slow_pos = _positions[min_point]
        curve_ordered = np.hstack((curve[fast_pos], curve[slow_pos], curve[-1]))
        return curve_ordered
    except:
      return np.array([np.nan,np.nan,np.nan,np.nan,np.nan])
def find_downward(dataI):
    diff_I = np.diff(dataI)
    downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
    return downwardinfl

def exp_decay_factor(dataT,dataV,dataI, time_aft, abf_id='abf', plot=False, root_fold=''):
     try:
        time_aft = time_aft / 100
        if time_aft > 1:
            time_aft = 1

        diff_I = np.diff(dataI)
        downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
        end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
        
        upperC = np.amax(dataV[downwardinfl:end_index])
        lowerC = np.amin(dataV[downwardinfl:end_index])
        diff = np.abs(upperC - lowerC)
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        SpanFast=(upperC-lowerC)*1*.01
        curve, pcov_2p = curve_fit(exp_decay_2p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=([-np.inf,  0, 0.1,  0, 0], [np.inf, np.inf, 500, np.inf, np.inf]), xtol=None)
        curve2, pcov_1p = curve_fit(exp_decay_1p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=(-np.inf, np.inf))
        residuals_2p = dataV[downwardinfl:end_index]- exp_decay_2p(t1, *curve)
        residuals_1p = dataV[downwardinfl:end_index]- exp_decay_1p(t1, *curve2)
        ss_res_2p = np.sum(residuals_2p**2)
        ss_res_1p = np.sum(residuals_1p**2)
        ss_tot = np.sum((dataV[downwardinfl:end_index]-np.mean(dataV[downwardinfl:end_index]))**2)
        r_squared_2p = 1 - (ss_res_2p / ss_tot)
        r_squared_1p = 1 - (ss_res_1p / ss_tot)
        if plot == True:

            plt.figure(2)
            plt.clf()
            plt.plot(t1, dataV[downwardinfl:end_index], label='Data')
            plt.plot(t1, exp_decay_2p(t1, *curve), label='2 phase fit')
            plt.plot(t1, exp_decay_1p(t1, curve[0], curve[3]/4, curve[2]) + np.abs(upperC - np.amax(exp_decay_1p(t1, curve[0], curve[3]/4, curve[2]))), label='Phase 1', zorder=9999)
            plt.plot(t1, exp_decay_1p(t1, curve[0], curve[3], curve[4]) + np.abs(upperC - np.amax(exp_decay_1p(t1, curve[0], curve[3], curve[4]))), label='Phase 2')
            plt.legend()
            plt.title(abf_id)
            plt.pause(3)
            plt.savefig(root_fold+ '//cm_plots//' + abf_id+'.png')
            #plt.close() 
        tau1 = 1/curve[2]
        tau2 = 1/curve[4]
        tau_1p = 1/curve2[2]
        fast = np.argmin([tau1, tau2])
        slow = np.argmax([tau1, tau2])
        curve_ordered = []
        return [tau1, tau2][fast], [tau1, tau2][slow], curve, r_squared_2p, r_squared_1p, tau_1p
     except:
        return np.nan, np.nan, np.array([np.nan,np.nan,np.nan,np.nan,np.nan]), np.nan, np.nan, np.nan


def exp_decay_factor_alt(dataT,dataV,dataI, time_aft, abf_id='abf', plot=False, root_fold=''):
     try:
        time_aft = time_aft / 100
        if time_aft > 1:
            time_aft = 1

        diff_I = np.diff(dataI)
        downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
        
        end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
        upperC = np.amax(dataV[downwardinfl:end_index])
        lowerC = np.amin(dataV[downwardinfl:end_index])
        minpoint = np.argmin(dataV[downwardinfl:end_index])
        end_index = downwardinfl + int(.95 * minpoint)
        downwardinfl = downwardinfl #+ int(.10 * minpoint)

        diff = (np.abs(upperC - lowerC) / 4) / 1000
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        SpanFast=(upperC-lowerC)*1*.01
       # print(f"One phase decay fit")
        tstart_ = time.time()
        curve2, pcov_1p = curve_fit(exp_decay_1p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=([-np.inf, -np.inf, 0], np.inf))
        #print(f"One phase decay fit end {(time.time() - tstart_)/60}")
        tstart_ = time.time()
        #print(f"two phase decay fit")
        curve, pcov_2p = curve_fit(exp_decay_2p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=([-np.inf, -np.inf, 0,  -np.inf, 0], [np.inf, 0, curve2[2], 0, 0.01]), verbose=1, xtol=None)
       # print(f"two phase decay fit end {(time.time() - tstart_)/60}")
        residuals_2p = dataV[downwardinfl:end_index]- exp_decay_2p(t1, *curve)
        residuals_1p = dataV[downwardinfl:end_index]- exp_decay_1p(t1, *curve2)
        ss_res_2p = np.sum(residuals_2p**2)
        ss_res_1p = np.sum(residuals_1p**2)
        ss_tot = np.sum((dataV[downwardinfl:end_index]-np.mean(dataV[downwardinfl:end_index]))**2)
        r_squared_2p = 1 - (ss_res_2p / ss_tot)
        r_squared_1p = 1 - (ss_res_1p / ss_tot)
        if plot == True:
            try:
                end_index2 = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
                t1 = dataT[downwardinfl:end_index2] - dataT[downwardinfl]
                plt.figure(2)
                plt.clf()
                plt.plot(t1, dataV[downwardinfl:end_index2], label='Data')
                plt.plot(t1, exp_decay_2p(t1, *curve), label='2 phase fit')
                plt.plot(t1, exp_decay_1p(t1, curve[0], curve[1], curve[2]), label='Phase 1', zorder=9999)
                plt.plot(t1, exp_decay_1p(t1, curve[0], curve[3], curve[4]), label='Phase 2')
                plt.legend()
                plt.title(abf_id)
                plt.pause(3)
                plt.savefig(root_fold+ '//cm_plots//' + abf_id+'.png')
            except:
                pass
            #plt.close() 
        tau1 = curve[2]
        tau2 = curve[4]
        tau_1p = curve2[2]
        fast = np.argmin([tau1, tau2])
        slow = np.argmax([tau1, tau2])
        _positions = [[1,2], [3,4]]
        fast_pos = _positions[fast]
        slow_pos = _positions[slow]
        curve_ordered = np.hstack((curve[0], curve[fast_pos], curve[slow_pos]))
        
        return [tau1, tau2][fast], [tau1, tau2][slow], curve_ordered, r_squared_2p, r_squared_1p, tau_1p
     except:
        return np.nan, np.nan, np.array([np.nan,np.nan,np.nan,np.nan,np.nan]), np.nan, np.nan, np.nan

def compute_sag(dataT,dataV,dataI, time_aft, plot=False):
   try:
         time_aft = time_aft / 100
         if time_aft > 1:
                time_aft = 1   
         diff_I = np.diff(dataI)
         upwardinfl = np.nonzero(np.where(diff_I>0, diff_I, 0))[0][0]
         test = dataT[upwardinfl]
         diff_I = np.diff(dataI)
         downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
         dt = dataT[1] - dataT[0] #in s
         end_index = upwardinfl - int(0.100/dt)
         end_index2 = upwardinfl - int((upwardinfl - downwardinfl) * time_aft)
         if end_index<downwardinfl:
             end_index = upwardinfl - 5
         vm = np.nanmean(dataV[end_index:upwardinfl])
         
         min_point = downwardinfl + np.argmin(dataV[downwardinfl:end_index2])
         test = dataT[downwardinfl]
         test2 = dataT[end_index]
         avg_min = np.nanmean(dataV[min_point])
         sag_diff = avg_min - vm
         sag_diff_plot = np.arange(avg_min, vm, 1)
         if plot==True:
             try:
                 plt.figure(num=99)
                 plt.clf()
                 plt.plot(dataT[downwardinfl:int(upwardinfl+1000)], dataV[downwardinfl:int(upwardinfl + 1000)], label="Data")
                 plt.scatter(dataT[min_point], dataV[min_point], c='r', marker='x', zorder=99, label="Min Point")
                 plt.scatter(dataT[end_index:upwardinfl], dataV[end_index:upwardinfl], c='g', zorder=99, label="Mean Vm Measured")
                 plt.plot(dataT[np.full(sag_diff_plot.shape[0], min_point, dtype=np.int64)], sag_diff_plot, label=f"Sag of {sag_diff}")
                 plt.legend()
                 plt.pause(0.05)
             except:
                 print("plot fail")
         
         return sag_diff, avg_min
   except:
         return np.nan
        



def membrane_resistance(dataT,dataV,dataI):
    try:
        diff_I = np.diff(dataI)
        downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
        end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl)/2)
        
        upperC = np.mean(dataV[:downwardinfl-100])
        lowerC = np.mean(dataV[downwardinfl+100:end_index-100])
        diff = -1 * np.abs(upperC - lowerC)
        I_lower = dataI[downwardinfl+1]
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        #v = IR
        #r = v/I
        v_ = diff / 1000 # in mv -> V
        I_ = I_lower / 1000000000000 #in pA -> A
        r = v_/I_

        return r #in ohms
    except: 
        return np.nan

def mem_cap(resist, tau_2p, tau_1p=np.nan):
    #tau = RC
    #C = R/tau
    C_2p = tau_2p / resist
    C_1p = tau_1p / resist
    return C_2p, C_1p ##In farads?


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
proto = input("enter Protocol to analyze (enter -1 to not filter any protocol): ")
try: 
    proto = int(proto)
except:
    proto = -1
if proto == -1:
    protocol_name = 'all'
else:
    protocol_name = protocol_n[proto]


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
        
def plotabf(abf, spiketimes, lowerlim, upperlim, sweep_plots):
   try:
    if sweep_plots[0] == -1:
        pass
    else:
        plt.figure(num=2, figsize=(16,6))
        plt.clf()
        cm = plt.get_cmap("Set1") #Changes colour based on sweep number
        if sweep_plots[0] == 0:
            sweepList = abf.sweepList
        else:
            sweepList = sweep_plots - 1
        colors = [cm(x/np.asarray(sweepList).shape[0]) for x,_ in enumerate(sweepList)]
        
        plt.autoscale(True)
        plt.grid(alpha=0)

        plt.xlabel(abf.sweepLabelX)
        plt.ylabel(abf.sweepLabelY)
        plt.title(abf.abfID)

        for c, sweepNumber in enumerate(sweepList):
            abf.setSweep(sweepNumber)
            
            spike_in_sweep = (spiketimes[spiketimes[:,1]==int(sweepNumber+1)])[:,0]
            i1, i2 = int(abf.dataRate * lowerlim), int(abf.dataRate * upperlim) # plot part of the sweep
            dataX = abf.sweepX
            dataY = abf.sweepY
            colour = colors[c]
            sweepname = 'Sweep ' + str(sweepNumber)
            plt.plot(dataX, dataY, color=colour, alpha=1, lw=1, label=sweepname)
            
            plt.scatter(dataX[spike_in_sweep[:]], dataY[spike_in_sweep[:]], color=colour, marker='x')
           
        

        plt.xlim(abf.sweepX[i1], abf.sweepX[i2])
        plt.legend()
        
        plt.savefig(abf.abfID +'.png', dpi=600)
        plt.pause(0.05)
   except:
        print('plot failed')

def build_running_bin(array, time, start, end, bin=20, time_units='s', kind='nearest'):
    if time_units == 's':
        start = start * 1000
        end = end* 1000

        time = time*1000
    time_bins = np.arange(start, end+bin, bin)
    binned_ = np.full(time_bins.shape[0], np.nan, dtype=np.float64)
    index_ = np.digitize(time, time_bins)
    uni_index_ = np.unique(index_)
    for time_ind in uni_index_:
        data = np.asarray(array[index_==time_ind])
        data = np.nanmean(data)
        binned_[time_ind] = data
    nans = np.isnan(binned_)
    if np.any(nans):
        if time.shape[0] > 1:
            f = interpolate.interp1d(time, array, kind=kind, fill_value="extrapolate")
            new_data = f(time_bins)
            binned_[nans] = new_data[nans]
        else:
            binned_[nans] = np.nanmean(array)
    return binned_, time_bins

def mem_cap_alt(tau, b2, deflection):
    rm2 = np.abs((b2)/(deflection /1000000000000))#in pA -> A)
    cm = tau / rm2
    return cm

def determine_subt(abf):
    def nonzero_1d(a):
        non = np.nonzero(a)
        return a[non]
    dataC =[]
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        sweepdiff = np.diff(abf.sweepC)
        dataC.append(sweepdiff)
    dataC = np.vstack(dataC)
    deflections = np.unique(np.where(dataC<0)[0])
    return deflections


debugplot = 0
running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width']
dfs = pd.DataFrame()
averages = pd.DataFrame()
for root,dir,fileList in os.walk(files):
 for filename in fileList:
    if filename.endswith(".abf"):
        file_path = os.path.join(root,filename)
        try:
            abf = pyabf.ABF(file_path)
        
            if proto == -1 or (abf.sweepLabelY != 'Clamp Current (pA)' and abf.protocol != 'Gap free' and protocol_name in abf.protocol):
                print(filename + ' import')
                
              #try:
                np.nan_to_num(abf.data, nan=-9999, copy=False)
                if savfilter >0:
                    abf.data = signal.savgol_filter(abf.data, savfilter, polyorder=3)
                
                 #If there is more than one sweep, we need to ensure we dont iterate out of range
                if abf.sweepCount > 1:
                    if subt_sweeps is None:
                        sweepList = determine_subt(abf)
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

                    print(f"Fitting to sweep {real_sweep_number}")

                    abf.setSweep(sweepNumber)
                    
                    dataT, dataV, dataI = abf.sweepX, abf.sweepY, abf.sweepC
                    decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, p_decay = exp_decay_factor(dataT, dataV, dataI, time_after, abf_id=abf.abfID, plot=False)
                    phase2_amp = min(curve[1],curve[3])
                    resist = membrane_resistance(dataT, dataV, dataI)
                    Cm2, Cm1 = mem_cap(resist, decay_slow, p_decay)
                    Cm3 = mem_cap_alt(decay_slow, phase2_amp, np.amin(dataI))
                    #grow = exp_growth_factor(dataT, dataV, dataI, decay_slow, plot=True)
                    #grow = np.full(6, np.nan)
                    
                    temp_df[f"_1 phase decay {real_sweep_number}"] = [p_decay]           
                    temp_df[f"fast 2 phase decay {real_sweep_number}"] = [decay_fast]
                    temp_df[f"slow 2 phase decay {real_sweep_number}"] = [decay_slow]
                    temp_df[f"Curve fit A {real_sweep_number}"] = [curve[0]]
                    temp_df[f"Curve fit b1 {real_sweep_number}"] = [curve[1]]
                    temp_df[f"Curve fit b2 {real_sweep_number}"] = [curve[3]]
                    temp_df[f"R squared 2 phase {real_sweep_number}"] = [r_squared_2p]
                    temp_df[f"R squared 1 phase {real_sweep_number}"] = [r_squared_1p]
                    #temp_df[f"Growth fit fast tau {real_sweep_number}"] = [grow[2]]
                    #temp_df[f"Growth fit slow tau {real_sweep_number}"] = [grow[4]]
                    temp_df[f"RMP {real_sweep_number}"] = [rmp_mode(dataV, dataI)]
                    temp_df[f"Membrane Resist {real_sweep_number}"] =  resist / 1000000000 #to gigaohms
                    temp_df[f"_2 phase Cm {real_sweep_number}"] =  Cm2 * 1000000000000#to pf farad
                    temp_df[f"_ALT_2 phase Cm {real_sweep_number}"] =  Cm3 * 1000000000000
                    temp_df[f"_1 phase Cm {real_sweep_number}"] =  Cm1 * 1000000000000
                    temp_df[f"Voltage sag {real_sweep_number}"], _ = compute_sag(dataT,dataV,dataI, time_after)
                    sag_ratio, taum_allen, voltage_allen = subthres_a(dataT,dataV,dataI, 0, np.amax(dataT))
                    temp_df[f"Voltage sag ratio {real_sweep_number}"] = sag_ratio
                    temp_df[f"Tau_m Allen {real_sweep_number}"] = taum_allen    
                    temp_df[f"Voltage sag Allen {real_sweep_number}"] = voltage_allen[0]
                    full_dataI.append(dataI)
                    full_dataV.append(dataV)
                    if dataI.shape[0] < dataV.shape[0]:
                        dataI = np.hstack((dataI, np.full(dataV.shape[0] - dataI.shape[0], 0)))
                

                
                full_dataI = np.vstack(full_dataI) 
                #indices_of_same = np.any(full_dataI, axis=1)
                #if np.any(indices_of_same) == False:
                #    indices_of_same = 0
                indices_of_same = np.arange(full_dataI.shape[0])
                full_dataV = np.vstack(full_dataV)
                if bplot == True:
                    if not os.path.exists(root_fold+'//cm_plots//'):
                            os.mkdir(root_fold+'//cm_plots//')   
                print("Fitting Decay")
                decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, p_decay = exp_decay_factor_alt(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0), time_after, abf_id=abf.abfID, plot=bplot, root_fold=root_fold)
                print("Computing Sag")
                grow = exp_growth_factor(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0), decay_slow)
                
                temp_avg[f"Voltage sag mean"], temp_avg[f"Voltage sag min point"] = compute_sag(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0), time_after, plot=bplot)
                if bplot == True:
                    plt.title(abf.abfID)
                    plt.savefig(root_fold+'//cm_plots//sagfit'+abf.abfID)
                phase2_amp = min(curve[1],curve[3])
                temp_avg["Averaged 1 phase decay "] = [p_decay]           
                temp_avg["Averaged 2 phase fast decay "] = [decay_fast]
                temp_avg["Averaged 2 phase slow decay "] = [decay_slow]
                temp_avg["Averaged Curve fit A"] = [curve[0]]
                temp_avg["Averaged Curve fit b1"] = [curve[1]]
                temp_avg["Averaged Curve fit b2"] = [curve[3]]
                temp_avg["Averaged R squared 2 phase"] = [r_squared_2p]
                temp_avg["Averaged R squared 1 phase"] = [r_squared_1p]
                temp_avg[f"Averaged RMP"] = [rmp_mode(np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0))]
                temp_avg["SweepCount Measured"] = [sweepcount]
                temp_avg["Averaged Grow Fast tau"] = [grow[2]]
                temp_avg["Averaged Grow slow tau"] = [grow[4]]
                if r_squared_2p > r_squared_1p:
                   temp_avg["Averaged Best Fit"] = [2]
                else:
                    temp_avg["AverageD Best Fit"] = [1]
                print(f"fitting Membrane resist")
                resist = membrane_resistance(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0))
                resist_alt = exp_rm_factor(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0), np.nanmean(full_dataI[indices_of_same,:],axis=0), time_after, p_decay, abf_id=abf.abfID,  root_fold=root_fold, plot=bplot)
                Cm2, Cm1 = mem_cap(resist, decay_slow, p_decay)
                Cm3 = mem_cap_alt(decay_slow, phase2_amp, np.amin(np.nanmean(full_dataI[indices_of_same,:],axis=0)))
                rm_alt = mem_resist_alt(Cm3, decay_slow)
                temp_avg["Averaged Membrane Resist"] =  resist  / 1000000000 #to gigaohms
                temp_avg["Averaged Membrane Resist_ALT"] =  resist_alt[0]  / 1000000000
                temp_avg["Averaged Membrane Resist_ALT 2"] =  rm_alt  / 1000000000#to gigaohms
                temp_avg["Averaged Pipette Resist_ALT"] =  resist_alt[2]  / 1000000000 #to gigaohms
                temp_avg["Averaged 2 phase Cm"] =  Cm2 * 1000000000000
                temp_avg["Averaged 2 phase Cm Alt"] =  Cm3 * 1000000000000
                temp_avg["Averaged 1 phase Cm"] =  Cm1 * 1000000000000
                sag_ratio, taum_allen, voltage_allen = subthres_a(dataT, np.nanmean(full_dataV[indices_of_same,:],axis=0),
                                                                   np.nanmean(full_dataI[indices_of_same,:],axis=0),0.0, np.amax(dataT))
                temp_avg[f"Averaged Voltage sag ratio {real_sweep_number}"] = sag_ratio
                temp_avg[f"Averaged Tau_m Allen {real_sweep_number}"] = taum_allen    
                temp_avg[f"Average Voltage sag Allen {real_sweep_number}"] = voltage_allen
                temp_avg["Protocol"] =  abf.protocol
                print(f"Computed a membrane resistance of {(resist  / 1000000000)} giga ohms, and a capatiance of {Cm2 * 1000000000000} pF, and tau of {decay_slow*1000} ms")
                dfs = dfs.append(temp_df, sort=True)
                averages = averages.append(temp_avg, sort=True)
              #except:
               # print('Issue Processing ' + filename)

            else:
                print('Not correct protocol: ' + abf.protocol)
        except:
          print('Issue Processing ' + filename)


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