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
from ipfx import feature_extractor as fx
from . import patch_utils
from . import utils
import pyabf
from . import loadFile
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
        t1 = dataT[upwardinfl:end_index] - dataT[upwardinfl]
        curve = curve_fit(exp_grow, t1, dataV[upwardinfl:end_index], maxfev=50000, bounds=([-np.inf, -np.inf, alpha-0.05], [np.inf, np.inf, alpha+0.05]), xtol=None)[0]
        
        _, deriv_ar = deriv(t1, exp_grow(t1, *curve))
        diff = np.abs(deriv_ar - 2)
        minpoint = np.argmin(diff)
        if plot==True:
            
            plt.figure(2)
            plt.clf()
            plt.plot( dataT[upwardinfl:end_index], dataV[upwardinfl:end_index], label='Data')
            plt.scatter(t1[minpoint], exp_grow(t1, *curve)[minpoint], label='min')
            plt.plot( dataT[upwardinfl:end_index], exp_grow(t1, *curve), label='1 phase fit')
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
    try:
        pre = find_downward(dataI)
        mode_vm = mode(np.round(dataV[:pre]*4)/4, nan_policy='omit')[0][0]
        return mode_vm
    except:
        return np.nan

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

@utils.debug_wrap
def exp_decay_factor(dataT,dataV,dataI, time_aft, abf_id='abf', plot=False, root_fold='') -> tuple[float, float, tuple[float, float, float, float, float], float, float, float]:
    
    time_aft = time_aft / 100
    if time_aft > 1:
        time_aft = 1

    diff_I = np.diff(dataI)
    downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
    end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)
    
    if end_index - downwardinfl < 10:
        return np.nan, np.nan, np.array([np.nan,np.nan,np.nan,np.nan,np.nan]), np.nan, np.nan, np.nan
    upperC = np.amax(dataV[downwardinfl:end_index])
    lowerC = np.amin(dataV[downwardinfl:end_index])
    diff = np.abs(upperC - lowerC)
    t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
    SpanFast=(upperC-lowerC)*1*.01

    curve2, pcov_1p = curve_fit(exp_decay_1p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=(-np.inf, np.inf))

    p_guess = (curve2[0], curve2[1], curve2[-1]*0.1, curve2[1], curve2[-1])
    curve, pcov_2p = curve_fit(exp_decay_2p, t1, dataV[downwardinfl:end_index], p0=p_guess,
                               maxfev=50000,  bounds=([-np.inf,  0, (1/500),  0, 0], [np.inf, np.inf, 500, np.inf, np.inf]),
                                 xtol=None)
    
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

        guess = (lowerC, diff, 50)
        curve2, pcov_1p = curve_fit(exp_decay_1p, t1, dataV[downwardinfl:end_index], maxfev=50000, p0=guess,
                                      bounds=([-np.inf, diff-15, 2], [np.inf, diff+15, 750]), xtol=None)
        curve, pcov_2p = curve_fit(exp_decay_2p, t1, dataV[downwardinfl:end_index], maxfev=50000,  bounds=([-np.inf,  0, 100,  0, 0], [np.inf, np.inf, 500, np.inf, np.inf]), xtol=None)
        
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
    def flatten_stf(lis):
        for item in lis:
            if isinstance(item, list):
                for subitem in flatten_stf(item):
                    yield subitem
            elif isinstance(item, dict):
                for subitem in flatten_stf(item.values()):
                    yield subitem
            else:
                yield item
    string_to_find = list(flatten_stf(string_to_find))
    #flatten the string to find (should be a lsit of strings, but some are dicts)
    columns = df.columns.values
    out = []
    for col in columns:
        string_found = [x in col for x in string_to_find]
        if np.any(string_found):
            out.append(col)
    return df[out]


@utils.debug_wrap
def compute_sag(dataT,dataV,dataI, time_aft, plot=False, clear=True) -> tuple[float, float]:
    """
    Computes the sag amplitude during a hyperpolarizing current injection.
    Takes:
    :param dataT: Description
    :param dataV: Description
    :param dataI: Description
    :param time_aft: Description
    :param plot: Description
    :param clear: Description
    :return: Description
    :rtype: tuple[float, float]
    """
    time_aft = time_aft / 100 #convert to proportion
    if time_aft > 1:
        time_aft = 1   #convert to proportion
    diff_I = np.diff(dataI) # find the points where current changes
    upwardinfl = np.nonzero(np.where(diff_I>0, diff_I, 0))[0][0] #first point where current goes up
    downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0] #first point where current goes down
    dt = dataT[1] - dataT[0] #in s #time difference between points
    end_index = upwardinfl - int(0.100/dt) #calculate end index based on 100ms before end of pulse
    end_index2 = upwardinfl - int((upwardinfl - downwardinfl) * time_aft) #calculate end index based on time after
    if end_index<downwardinfl:
        end_index = upwardinfl - 5 #ensure we have at least some points
    vm = np.nanmean(dataV[end_index:upwardinfl])
    
    min_point = downwardinfl + np.argmin(dataV[downwardinfl:end_index2]) #index of the min point
    avg_min = np.nanmean(dataV[min_point]) #average of the min point
    sag_diff = avg_min - vm #sag amplitude
    #sag_diff_plot = np.arange(avg_min, vm, 1)
    #plotting code removed for clarity
    return sag_diff, avg_min
   
        


@utils.debug_wrap
def membrane_resistance(dataT,dataV,dataI) -> float:
    """
    Computes the membrane resistance using the hyperpolarization segment of the current injection. (Assumes a square pulse)
    
    :param dataT: Description
    :param dataV: Description
    :param dataI: Description
    :return: Description
    :rtype: float
    """
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

def determine_subt(abf, idx_bounds, filter_spikes=False):
    """Determine which sweeps are subthreshold based on the current injection.
    Args:
        abf (_type_): _description_
        idx_bounds (_type_): _description_
        filter_spikes (bool, optional): _description_. Defaults to False.
    Returns:
        _type_: _description_
    """
    #for compat reasons this function needs to open the abf file itself. ideally this should be done in the main script, but for now we will do it here.
    if isinstance(abf, str):
        abf = loadFile.loadFile(abf)
    elif isinstance(abf, pyabf.ABF):
        abf = loadFile.loadFile(abf.abfFilePath)

    abf = [x[:, idx_bounds[0]:idx_bounds[1]] for x in abf] #

    #find the sweepwise difference in current amp
    diff_I = np.diff(abf[2], axis=0)
    #filter out nonzero values
    ladder_pulse_point = [[]]
    for row in diff_I:
        #assuming sqaure pulse here, the nonzero idxs should be continous
        non_zero_idxs = np.flatnonzero(row)
        if len(non_zero_idxs) > 1:
            #compute the difference and assert that its 1
            diff_idx = np.diff(non_zero_idxs)
            #filter down to the first one that is not 1
            first_non_one = np.flatnonzero(diff_idx != 1)
            skip_idx = 0 #int(len(non_zero_idxs))
            if len(first_non_one) > 0:
                ladder_pulse_point.append(non_zero_idxs[skip_idx:first_non_one[0]] )
            else:
                ladder_pulse_point.append(non_zero_idxs[skip_idx:] )
        else:
            ladder_pulse_point.append([])
    if filter_spikes:
        spikefx = fx.SpikeFeatureExtractor(filter = 0)
        non_spike_sweeps = []
        for i, (sweepX, sweepY, sweepC) in enumerate(zip(abf[0], abf[1], abf[2])):
            try:
                if len(ladder_pulse_point[i]) > 0:
                    sweep_features= spikefx.process(sweepX[ladder_pulse_point[i]], sweepY[ladder_pulse_point[i]], sweepC[ladder_pulse_point[i]])
                    if sweep_features.empty:
                        non_spike_sweeps.append(i)
            except:
                pass
                #non_spike_sweeps.append(i)
    else:
        non_spike_sweeps = list(range(len(abf[0])))
    
    #if there are less than 2 non spike sweeps, we cant compute the membrane resistance
    if len(non_spike_sweeps) < 1:
        #logging.warning("No non-spiking sweeps found in the abf file. Returning empty list.")
        return []

    return non_spike_sweeps

def nonzero_1d(a):
    non = np.nonzero(a)
    return a[non]

def ladder_rm(dataT, dataV, dataI, mean_current=False):
    """ Computes the membrane resistance using the ladder method. Essentially we need a changing hyperpolarization / depolarization segment
    to compute the membrane resistance.
    """
    #find the sweepwise difference in current amp
    diff_I = np.diff(dataI, axis=0)
    #filter out nonzero values
    ladder_pulse_point = [[]]
    for row in diff_I:
        #assuming sqaure pulse here, the nonzero idxs should be continous
        non_zero_idxs = np.flatnonzero(row)
        if len(non_zero_idxs) > 1:
            #compute the difference and assert that its 1
            diff_idx = np.diff(non_zero_idxs)
            #filter down to the first one that is not 1
            first_non_one = np.flatnonzero(diff_idx != 1)
            #finally we want to adjust to skip the first 25% of the pulse
            skip_idx = int(len(non_zero_idxs) * 0.25)
            if len(first_non_one) > 0:
                ladder_pulse_point.append(non_zero_idxs[skip_idx:first_non_one[0]] )
            else:
                ladder_pulse_point.append(non_zero_idxs[skip_idx:] )
        else:
            ladder_pulse_point.append([])
    # we need to find the spiking sweeps, and drop them
    #find the spiking sweeps
    spikefx = fx.SpikeFeatureExtractor(filter = 0)
    non_spike_sweeps = []
    for i, (sweepX, sweepY, sweepC) in enumerate(zip(dataT, dataV, dataI)):
        try:
            if len(ladder_pulse_point[i]) > 0:
                sweep_features= spikefx.process(sweepX[ladder_pulse_point[i]], sweepY[ladder_pulse_point[i]], sweepC[ladder_pulse_point[i]])
                if sweep_features.empty:
                    non_spike_sweeps.append(i)
        except:
            pass
            #non_spike_sweeps.append(i)
    
    #if there are less than 2 non spike sweeps, we cant compute the membrane resistance
    if len(non_spike_sweeps) < 2:
        return np.nan
    
    #finally fit a line to the ladder pulse point by taking the means
    if mean_current:
        sweep_mean_V = []
        sweep_mean_I = []
        for i in non_spike_sweeps:
            sweep_mean_V.append(np.mean(dataV[i][ladder_pulse_point[i]]))
            sweep_mean_I.append(np.mean(dataI[i][ladder_pulse_point[i]]))
    else:
        sweep_mean_V = np.ravel([dataV[i][ladder_pulse_point[i]] for i in non_spike_sweeps])
        sweep_mean_I = np.ravel([dataI[i][ladder_pulse_point[i]] for i in non_spike_sweeps])
    #fit a line to the mean V and I
    slope, intercept = np.polyfit(sweep_mean_V, sweep_mean_I, 1)
    return slope, intercept, len(non_spike_sweeps)



def subthres_a(dataT, dataV, dataI, lowerlim, upperlim):
    """Analyze the subthreshold features of the current using allen institute's method.

    Args:
        dataT (_type_): _description_
        dataV (_type_): _description_
        dataI (_type_): _description_
        lowerlim (_type_): _description_
        upperlim (_type_): _description_

    Returns:
        _type_: _description_
    """
    if dataI[np.argmin(dataI)] < 0: #if the current is negative check
                        try:
                            if lowerlim < 0.1:
                                b_lowerlim = 0.1
                            else:
                                b_lowerlim = 0.1

                            #get only the hyperpor segments
                            dwninf, upinf = find_hyperpolarization_segment(dataT, dataI, lowerlim, upperlim)
                            lowerlim_t = np.clip(dataT[dwninf]  - 0.1, 0, 1e9)
                            upperlim_t = dataT[upinf]

                            #temp_spike_df['baseline voltage' + real_sweep_number] = subt.baseline_voltage(dataT, dataV, start=b_lowerlim)
                            sag = subt.sag(dataT,dataV,dataI, start=lowerlim_t, end=upperlim_t)
                            taum = subt.time_constant(dataT,dataV,dataI, start=lowerlim_t, end=upperlim_t)
                            
                            voltage_deflection = subt.voltage_deflection(dataT,dataV,dataI, start=lowerlim_t, end=upperlim_t)
                            return sag, taum, voltage_deflection
                        except Exception as e:
                            print("Subthreshold Processing Error ")
                            print(e.args)
                            return np.nan, np.nan, [np.nan, np.nan]
    else:
        return np.nan, np.nan, [np.nan, np.nan]

def find_hyperpolarization_segment(dataT, dataI, lowerlim, upperlim):
    """Finds the hyperpolarization segment, assuming the current is a square pulse. Or the hyperpolarization is continuous.

    Args:
        dataT (_type_): _description_
        dataI (_type_): _description_
        lowerlim (_type_): _description_
        upperlim (_type_): _description_
    """
    #copy dataI so we dont change the original
    dataI = dataI.copy()
    #clip greater than 0 to 0
    dataI[dataI>0] = 0
    #find the first point where the current is negative
    downwardinfl = np.nonzero(np.where(dataI<0, dataI, 0))[0][0]
    #find the last point where the current is negative
    upwardinfl = np.nonzero(np.where(dataI<0, dataI, 0))[0][-1]
    return downwardinfl, upwardinfl
