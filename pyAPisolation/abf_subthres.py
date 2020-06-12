import sys
import numpy as np
from numpy import genfromtxt
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
from scipy.optimize import curve_fit
import pyabf




def exp_grow(t, a, b, alpha):
    return a - b * np.exp(-alpha * t)
def exp_decay_2p(t, a, b1, alphaFast, b2, alphaSlow):
    return a + b1*np.exp(-alphaFast*t) + b2*np.exp(-alphaSlow*t)
def exp_decay_1p(t, a, b1, alphaFast):
    return a + b1*np.exp(-alphaFast*t)
def exp_growth_factor(dataT,dataV,dataI, end_index=300):
    try:
        
        diff_I = np.diff(dataI)
        upwardinfl = np.argmax(diff_I)
        
        upperC = np.amax(dataV[upwardinfl:end_index])
        t1 = dataT[upwardinfl:end_index] - dataT[upwardinfl]
        curve = curve_fit(exp_grow, t1, dataV[upwardinfl:end_index], maxfev=50000, bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))[0]
        tau = curve[2]
        return 1/tau
    except:
        return np.nan


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