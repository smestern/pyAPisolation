import sys
import numpy as np
from numpy import genfromtxt
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.stats import mode
from ipfx import subthresh_features as subt
from . import patch_utils
import pyabf
#from brian2.units import ohm, Gohm, amp, volt, mV, second, pA


##Declare our options at default

def exp_grow(t, a, b, alpha):
    return a - b * np.exp(-alpha * t)
def exp_decay_2p(t, a, b1, alphaFast, b2, alphaSlow):
    return a + b1*np.exp(-alphaFast*t) + b2*np.exp(-alphaSlow*t)
def exp_decay_1p(t, a, b1, alphaFast):
    return a + b1*np.exp(-alphaFast*t)

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
        end_index += upwardinfl
        upperC = np.amax(dataV[upwardinfl:end_index])
        t1 = dataT[upwardinfl:end_index] - dataT[upwardinfl]
        curve = curve_fit(exp_grow, t1, dataV[upwardinfl:end_index], maxfev=50000, bounds=([-np.inf, -np.inf, alpha-0.05], [np.inf, np.inf, alpha+0.05]), xtol=None)[0]
        tau = curve[2]
        x_deriv, deriv_ar = deriv(t1, exp_grow(t1, *curve))
        diff = np.abs(deriv_ar - 2)
        minpoint = np.argmin(diff)
        if plot==True:
            
            plt.figure(2)
            plt.clf()
            plt.plot(t1, dataV[upwardinfl:end_index], label='Data')
            plt.scatter(t1[minpoint], exp_grow(t1, *curve)[minpoint], label='min')
            plt.plot(t1, exp_grow(t1, *curve), label='1 phase fit')
            plt.legend()
            #plt.title(abf_id)
            plt.pause(0.5)
            #plt.savefig(root_fold+ '//cm_plots//' + abf_id+'.png')
            #plt.close() 
        return np.hstack((curve, (t1[0]-t1[minpoint])))
    except:
        return np.array([np.nan, np.nan, np.nan, np.nan, np.nan])


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
        downwardinfl = downwardinfl + int(.10 * minpoint)
        
        upperC = np.amax(dataV[downwardinfl:end_index])
        lowerC = np.amin(dataV[downwardinfl:end_index])
        diff = upperC/1000 + -1* np.abs(upperC - lowerC) /1000 + -0.005
        Iinj_real = dataI[int(downwardinfl + 5)] / 1000000000000 #in pA -> A
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        SpanFast=(upperC-lowerC)*1*.01
        curve, pcov_2p = curve_fit(lambda t1, Rm, alphaFast, Re, alphaSlow, a: rm_decay_2p(t1, Iinj_real, Rm, alphaFast, Re, alphaSlow, a), t1, dataV[downwardinfl:end_index], maxfev=50000, bounds=([100000000, 0, 0, 0, (upperC /1000)-0.002], [np.inf, 1, 0.1, 0.01, (upperC /1000)+0.002]), verbose=1, xtol=None)

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
            #plt.close() 
        return curve
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
        fast = np.min([tau1, tau2])
        slow = np.max([tau1, tau2])
        return tau1, tau2, curve, r_squared_2p, r_squared_1p, tau_1p
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
        downwardinfl = downwardinfl + int(.10 * minpoint)

        diff = np.abs(upperC - lowerC) + 5
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        SpanFast=(upperC-lowerC)*1*.01
        curve, pcov_2p = curve_fit(exp_decay_2p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=([-np.inf,  0, 100,  0, 0], [np.inf, np.inf, 500, np.inf, np.inf]), xtol=None)
        curve2, pcov_1p = curve_fit(exp_decay_1p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=(-np.inf, np.inf))
        residuals_2p = dataV[downwardinfl:end_index]- exp_decay_2p(t1, *curve)
        residuals_1p = dataV[downwardinfl:end_index]- exp_decay_1p(t1, *curve2)
        ss_res_2p = np.sum(residuals_2p**2)
        ss_res_1p = np.sum(residuals_1p**2)
        ss_tot = np.sum((dataV[downwardinfl:end_index]-np.mean(dataV[downwardinfl:end_index]))**2)
        r_squared_2p = 1 - (ss_res_2p / ss_tot)
        r_squared_1p = 1 - (ss_res_1p / ss_tot)
        if plot == True:
            end_index2 = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
            t1 = dataT[downwardinfl:end_index2] - dataT[downwardinfl]
            plt.figure(2)
            plt.clf()
            plt.plot(t1, dataV[downwardinfl:end_index2], label='Data')
            plt.plot(t1, exp_decay_2p(t1, *curve), label='2 phase fit')
            plt.plot(t1, exp_decay_1p(t1, curve[0], curve[3]/4, curve[2]), label='Phase 1', zorder=9999)
            plt.plot(t1, exp_decay_1p(t1, curve[0], curve[3], curve[4]), label='Phase 2')
            plt.legend()
            plt.title(abf_id)
            plt.pause(3)
            plt.savefig(root_fold+ '//cm_plots//' + abf_id+'.png')
            #plt.close() 
        tau1 = 1/curve[2]
        tau2 = 1/curve[4]
        tau_1p = 1/curve2[2]
        fast = np.min([tau1, tau2])
        slow = np.max([tau1, tau2])
        return tau1, tau2, curve, r_squared_2p, r_squared_1p, tau_1p
     except:
        return np.nan, np.nan, np.array([np.nan,np.nan,np.nan,np.nan,np.nan]), np.nan, np.nan, np.nan

def df_select_by_col(df, string_to_find):
    columns = df.columns.values
    out = []
    for col in columns:
        string_found = [x in col for x in string_to_find]
        if np.any(string_found):
            out.append(col)
    return df[out]


def compute_sag(dataT,dataV,dataI, time_aft, plot=False, clear=True):
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
                 if clear:
                    plt.clf()
                 plt.plot(dataT[downwardinfl:int(upwardinfl+1000)], dataV[downwardinfl:int(upwardinfl + 1000)], label="Data")
                 plt.scatter(dataT[min_point], dataV[min_point], c='r', marker='x', zorder=99, label="Min Point")
                 plt.scatter(dataT[end_index:upwardinfl], dataV[end_index:upwardinfl], c='g', zorder=99, label="Mean Vm Measured")
                 plt.plot(dataT[np.full(sag_diff_plot.shape[0], min_point, dtype=np.int64)], sag_diff_plot, label=f"Sag of {sag_diff}")
                 #plt.legend()
                 plt.pause(0.05)
             except:
                 print("plot fail")
         
         return sag_diff, avg_min
   except:
         return np.nan, np.nan
        



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

def mem_cap(resist, tau_2p, tau_1p =np.nan):
    #tau = RC
    #C = R/tau
    C_2p = tau_2p / resist
    C_1p = tau_1p / resist
    return C_2p, C_1p ##In farads?

def mem_cap_alt(resist, tau, b2, deflection):
    rm2 = np.abs((b2/1000)/(deflection /1000000000000))#in pA -> A)
    cm = tau / rm2
    return cm

def determine_subt(abf, idx_bounds):
    def nonzero_1d(a):
        non = np.nonzero(a)
        return a[non]
    dataC =[]
    for sweep in abf.sweepList:
        abf.setSweep(sweep)
        sweepdiff = abf.sweepC[idx_bounds[0]:idx_bounds[1]]
        dataC.append(sweepdiff)
    dataC = np.vstack(dataC)
    deflections = np.unique(np.where(dataC<0)[0])
    return deflections


def subthres_a(dataT, dataV, dataI, lowerlim, upperlim):
    if dataI[np.argmin(dataI)] < 0:
                        try:
                            if lowerlim < 0.1:
                                b_lowerlim = 0.1
                            else:
                                b_lowerlim = 0.1
                            #temp_spike_df['baseline voltage' + real_sweep_number] = subt.baseline_voltage(dataT, dataV, start=b_lowerlim)
                            sag = subt.sag(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                            taum = subt.time_constant(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                            
                            voltage_deflection = subt.voltage_deflection(dataT,dataV,dataI, start=b_lowerlim, end=upperlim)
                            return sag, taum, voltage_deflection
                        except Exception as e:
                            print("Subthreshold Processing Error ")
                            print(e.args)
                            return np.nan, np.nan, np.nan
    else:
        return np.nan, np.nan, np.nan