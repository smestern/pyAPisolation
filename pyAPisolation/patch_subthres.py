import sys
import numpy as np
from numpy import genfromtxt
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
from scipy.optimize import curve_fit
import scipy.stats
from ipfx import subthresh_features as subt
from . import patch_utils
import pyabf
from brian2.units import ohm, Gohm, amp, volt, mV, second, pA



def exp_grow(t, a, b, alpha):
    return a - b * np.exp(-alpha * t)
def exp_grow_2p(t, a, b1, alphaFast, b2, alphaSlow):
    return a - b1 * np.exp(-alphaFast * t) - b2*np.exp(-alphaSlow*t) 


def exp_decay_2p(t, a, b1, alphaFast, b2, alphaSlow):
    return a + b1*np.exp(-alphaFast*t) + b2*np.exp(-alphaSlow*t)
def exp_decay_1p(t, a, b1, alphaFast):
    return a + b1*np.exp(-alphaFast*t)


def exp_growth_factor(dataT,dataV,dataI, end_index=300):
    try:
        
        diff_I = np.diff(dataI)
        upwardinfl = np.argmax(diff_I)

        #Compute out -50 ms from threshold
        dt = dataT[1] - dataT[0]
        offset = 0.05/ dt 

        end_index = int(end_index - offset)


        
        upperC = np.amax(dataV[upwardinfl:end_index])
        lowerC  = np.amin(dataV[upwardinfl:end_index])
        diffC = np.abs(lowerC - upperC) + 5
        t1 = dataT[upwardinfl:end_index] - dataT[upwardinfl]
        curve = curve_fit(exp_grow, t1, dataV[upwardinfl:end_index], maxfev=50000, bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))[0]
        curve2 = curve_fit(exp_grow_2p, t1, dataV[upwardinfl:end_index], maxfev=50000,   bounds=([-np.inf,  0, -np.inf,  0, -np.inf], [upperC + 5, diffC, np.inf, np.inf, np.inf]), xtol=None, method='trf')[0]
        tau = curve[2]
        #plt.plot(t1, dataV[upwardinfl:end_index])
        #plt.plot(t1, exp_grow_2p(t1, *curve2))
        #plt.title(f" CELL will tau1 {1/curve2[2]} and tau2 {1/curve2[4]}, a {curve2[0]} and b1 {curve2[1]}, b2 {curve2[3]}")
        #plt.pause(5)
        return curve2
    except:
        return [np.nan, np.nan, np.nan, np.nan, np.nan]

def exp_decay_factor(dataT,dataV,dataI, time_aft, abf_id='abf', plot=False, root_fold='', sag=True):
     try:
        time_aft = time_aft / 100
        if time_aft > 1:
            time_aft = 1

        if sag:
            diff_I = np.diff(dataI)
            downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
            
            end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
            upperC = np.amax(dataV[downwardinfl:end_index])
            lowerC = np.amin(dataV[downwardinfl:end_index])
            minpoint = np.argmin(dataV[downwardinfl:end_index])
            end_index = downwardinfl + int(.95 * minpoint)
            downwardinfl = downwardinfl #+ int(.10 * minpoint)
        else:
            diff_I = np.diff(dataI)
            downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
            end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
            
            upperC = np.amax(dataV[downwardinfl:end_index])
            lowerC = np.amin(dataV[downwardinfl:end_index])
        diff = np.abs(upperC - lowerC)
        t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
        SpanFast=(upperC-lowerC)*1*.01
        curve, pcov_2p = curve_fit(exp_decay_2p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=([-np.inf,  0, 100,  0, 0], [np.inf, np.inf, 500, np.inf, np.inf]))
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


def membrane_resistance(dataT,dataV,dataI):
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

def membrane_resistance_subt(dataT, dataV,dataI):
    resp_data = []
    stim_data = []
    for i, sweep in enumerate(dataV):
        abs_min, resp = compute_sag(dataT[i,:], sweep, dataI[i,:])
        ind = patch_utils.find_stim_changes(dataI[i, :])
        stim = dataI[i,ind[0] + 1]
        stim_data.append(stim)
        resp_data.append(resp+abs_min)
    resp_data = np.array(resp_data) * mV
    stim_data = np.array(stim_data) * pA
    res = scipy.stats.linregress(stim_data / amp, resp_data / volt)
    resist = res.slope * ohm
    return resist / Gohm

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

def compute_sag(dataT,dataV,dataI, time_aft=50):
         min_max = [np.argmin, np.argmax]
         find = 0
         time_aft = time_aft / 100
         if time_aft > 1:
                time_aft = 1   
         diff_I = np.diff(dataI)
         upwardinfl = np.nonzero(np.where(diff_I>0, diff_I, 0))[0][0]
         downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
         if upwardinfl < downwardinfl: #if its depolarizing then swap them
            temp = downwardinfl
            find = 1
            downwardinfl = upwardinfl
            upwardinfl = temp
         dt = dataT[1] - dataT[0] #in s
         end_index = upwardinfl - int(0.100/dt)
         end_index2 = upwardinfl - int((upwardinfl - downwardinfl) * time_aft)
         
         if end_index<downwardinfl:
             end_index = upwardinfl - 5
         vm = np.nanmean(dataV[end_index:upwardinfl])
         
         min_point = downwardinfl + min_max[find](dataV[downwardinfl:end_index2])
         test = dataT[downwardinfl]
         test2 = dataT[end_index]
         avg_min = np.nanmean(dataV[min_point])
         sag_diff = avg_min - vm

         return sag_diff, vm

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