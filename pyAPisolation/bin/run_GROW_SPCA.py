# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys
import numpy as np
from numpy import genfromtxt
import tkinter as tk
from tkinter import filedialog
import os
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import interpolate
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from scipy import stats
from ipfx import feature_extractor
from ipfx import subthresh_features as subt
from ipfx import feature_vectors as fv
from ipfx.sweep import Sweep
from sklearn.preprocessing import minmax_scale
from pyAPisolation.loadFile import loadABF
import sklearn.preprocessing
import pyabf
import logging
import glob
method='trf'


# import autograd.numpy as np
# from autograd import grad


def exp_grow(t, a, b, alpha):
    return a - b * np.exp(-alpha * t)

def exp_grow_2p(t, a, b1, alphaFast, b2, alphaSlow):
    return a - b1 * np.exp(-alphaFast * t) - b2*np.exp(-alphaSlow*t) 

def exp_grow_gp(t, Y0, Plateau, PercentFast, KFast, KSlow):
    #graphpad style two phase decay
    SpanFast=(Y0-Plateau)*PercentFast*.01

    SpanSlow=(Y0-Plateau)*(100-PercentFast)*.01

    return Plateau + SpanFast*np.exp(-KFast*t) + SpanSlow*np.exp(-KSlow*t)


# %%
def exp_growth_factor(dataT,dataV,dataI, end_index=300):
    #try:
        
        diff_I = np.diff(dataI)
        upwardinfl = np.argmax(diff_I)

        #Compute out -10 ms from threshold
        dt = dataT[1] - dataT[0]
        offset = 0.01/ dt 

        end_index = int(end_index - offset)
        if end_index <= upwardinfl:
            return [np.nan, np.nan, np.nan, np.nan, np.nan], np.nan
            #end_index = int(end_index)
        #if its still before the threshold, return nan
        if end_index <= upwardinfl:
            return [np.nan, np.nan, np.nan, np.nan, np.nan], np.nan


        
        upperC = np.amax(dataV[upwardinfl:end_index])
        lowerC  = np.amin(dataV[upwardinfl:end_index])
        diffC = np.abs(lowerC - upperC)
        t1 = dataT[upwardinfl:end_index] - dataT[upwardinfl]
        curve = curve_fit(exp_grow, t1, dataV[upwardinfl:end_index], maxfev=50000, bounds=([-np.inf, -np.inf, -np.inf], [np.inf, np.inf, np.inf]))[0]
        curve2 = curve_fit(exp_grow_2p, t1, dataV[upwardinfl:end_index], maxfev=50000,method='trf', bounds=([upperC-5,  0, 10,  0,  0], [upperC+5, diffC, np.inf, diffC,np.inf]), xtol=None, gtol=None, ftol=1e-12, jac='3-point')[0]
        tau = curve[2]
        tau1 = 1/curve2[2]
        tau2 = 1/curve2[4]
        tau_idx = [2, 4]
        fast = tau_idx[np.argmin([tau1, tau2])]
        slow = tau_idx[np.argmax([tau1, tau2])]
        
        curve_out = [curve2[0], curve2[fast-1], curve2[fast], curve2[slow-1], curve2[slow]]


        plt.subplot(1,2,1)
        plt.plot(t1, dataV[upwardinfl:end_index], c='k', alpha=0.5)
        plt.plot(t1, exp_grow_2p(t1, *curve2), label=f'2 phase fit', c='r', alpha=0.5)
        plt.plot(t1, exp_grow(t1, *curve_out[:3]), label=f'Fast phase', c='g', alpha=0.5)
        plt.plot(t1, exp_grow(t1, curve_out[0], *curve_out[3:]), label=f'slow phase', c='b', alpha=0.5)
        plt.title(f" CELL will tau1 {1/curve2[fast]} and tau2 {1/curve2[slow]}")
        #plt.subplot(1,2,2)
        plt.legend()
        plt.twinx()
        plt.subplot(1,2,2)
        #dy = curve_detrend(t1, dataV[upwardinfl:end_index], curve2)
        dy = signal.savgol_filter(np.diff(dataV[upwardinfl:end_index])/np.diff(t1*1000), 713, 2, mode='mirror')
        plt.plot(t1[:-1],dy)
        
        curve_out = [curve2[0], curve2[fast-1], 1/curve2[fast], curve2[slow-1], 1/curve2[slow]]
        return curve_out, np.amin(dy)
    #except:
        return [np.nan, np.nan, np.nan, np.nan, np.nan]


# %%


#f1 = grad(exp_grow_2p)  # 1st derivative of f
#f2 = grad(f1) # 2nd derivative of f

def curvature(x, a, b1, alphaFast, b2, alphaSlow):
        return np.abs(f2(x, a, b1, alphaFast, b2, alphaSlow))*(1 + f1(x, a, b1, alphaFast, b2, alphaSlow)**2)**-1.5
    
    
def curvature_real(dy, ddy):
        return abs(dy)*(1 + ddy**2)**-1.5
    
def curvature_splines(x, y=None, error=0.1, smoothing=None):
    """Calculate the signed curvature of a 2D curve at each point
    using interpolating splines.
    Parameters
    ----------
    x,y: numpy.array(dtype=float) shape (n_points, )
         or
         y=None and
         x is a numpy.array(dtype=complex) shape (n_points, )
         In the second case the curve is represented as a np.array
         of complex numbers.
    error : float
        The admisible error when interpolating the splines
    Returns
    -------
    curvature: numpy.array shape (n_points, )
    Note: This is 2-3x slower (1.8 ms for 2000 points) than `curvature_gradient`
    but more accurate, especially at the borders.
    """

    # handle list of complex case
    if y is None:
        x, y = x.real, x.imag

    t = np.arange(x.shape[0])
    std = error * np.ones_like(x)

    fx = UnivariateSpline(t, x, k=4, w=1 / np.sqrt(std), s=smoothing)
    fy = UnivariateSpline(t, y, k=4, w=1 / np.sqrt(std), s=smoothing)

    xˈ = fx.derivative(1)(t)
    xˈˈ = fx.derivative(2)(t)
    yˈ = fy.derivative(1)(t)
    yˈˈ = fy.derivative(2)(t)
    curvature = (xˈ* yˈˈ - yˈ* xˈˈ) / np.power(xˈ** 2 + yˈ** 2, 3 / 2)
    return curvature
    

def derivative(x,y):
    return np.diff(y)/np.diff(x)


def curve_detrend(x,y, curve2):
    test = curvature_splines(x, signal.savgol_filter(y, 51, 1), error=1, smoothing=25)
    cy = np.array([curvature(xi, *curve2) for xi in x])
    #detrend using first and last point
    lin_res = stats.linregress([x[0], x[-1]], [cy[0], cy[-1]])
    trend = x*lin_res.slope + lin_res.intercept
    #plt.plot(x,trend)
    detrended_data = cy - trend
    return detrended_data

def process_upwards_growth(in_file_path, out_file_path, IC1_file_path='/media/smestern/Expansion/PVN_MARM_PROJECT/IC1 Files_211117/', cell_type_column='cell_label'):
    files = glob.glob(IC1_file_path+'*.abf', recursive=True)

    if in_file_path is None:
        no_file_guide = True
        
    else:
        no_file_guide = False
        cell_type_df = pd.read_csv(in_file_path)
        print(cell_type_df.head())
        file_names = cell_type_df['filename'].to_numpy()
        cell_type_label = cell_type_df[cell_type_column].to_numpy()


    curves = []
    label = []
    ids = []
    max_curve = []
    sweepwise_data = {}
    for i, f in enumerate(files[:]):
        #print(i)
        try:
            base = os.path.basename(f)
            base = base.split(".")[0]
            
            if no_file_guide or base in file_names:
                x, y, c = loadABF(f)
                temp_min = []
                temp_curves =[]
                #plt.clf()
                iterd = 0
                for sweepX, sweepY, sweepC in zip(x,y,c):
                    spikext = feature_extractor.SpikeFeatureExtractor(filter=0, end=1.25)
                    res = spikext.process(sweepX, sweepY, sweepC)
                    if res.empty==False:
                        iterd += 1
                        spike_time = res['threshold_index'].to_numpy()[0]
                        #plt.figure(num=2)
                        curve, max_dy = exp_growth_factor(sweepX, sweepY, sweepC, spike_time)
                        temp_min.append(max_dy)
                        temp_curves.append(curve)
                temp_curves = np.vstack(temp_curves)
                div = np.ravel((temp_curves[:,2]) / (temp_curves[:,4])).reshape(-1,1)
                
                sum_height= (temp_curves[:,1] + temp_curves[:,3])
                ratio = (temp_curves[:,2] / (temp_curves[:,1] / sum_height)) / (temp_curves[:,4] / (temp_curves[:,3] / sum_height))
                ratio = np.ravel(ratio).reshape(-1,1)
                temp_curves = np.hstack([temp_curves, div, ratio])
                print(temp_curves)
                meanC = np.nanmean(temp_curves, axis=0)
                print(meanC.shape)
                curves.append(meanC)
                label_idx = np.argwhere(file_names==base) if not no_file_guide else 0
                
                max_curve.append(np.nanmean(temp_min))
                label.append(cell_type_label[label_idx] if not no_file_guide else 0) 
                
                ids.append(base)
                sweepwise_dict = {}
                for j, c in enumerate(temp_curves):
                    sweepwise_dict[f"plateau_{j}"] = c[0]
                    sweepwise_dict[f"perfast_{j}"] = c[1]
                    sweepwise_dict[f"taufast_{j}"] = c[2]
                    sweepwise_dict[f"perslow_{j}"] = c[3]
                    sweepwise_dict[f"tauslow_{j}"] = c[4]
                    sweepwise_dict[f"div_s_{j}"] = c[5]
                    sweepwise_dict[f"ratio_s_{j}"] = c[6]
                    
                    sweepwise_dict[f'max_dydt_{j}'] = temp_min[j]
                sweepwise_dict['filename'] = base
                sweepwise_data[base] = sweepwise_dict
                plt.savefig(f+".png")
                plt.pause(0.1)
                
                plt.close()
        except:
           print("fail")

    curves = np.vstack(curves)
    print(curves)
    label = np.ravel([x[0] for x in label]).reshape(-1,1)
    div = np.ravel((curves[:,2]) / (curves[:,4])).reshape(-1,1)
    print(div)
    sum_height= (curves[:,1] + curves[:,3])
    ratio = (curves[:,2] / (curves[:,1]/sum_height)) / (curves[:,4] / (curves[:,3]/sum_height))
    ratio = np.ravel(ratio).reshape(-1,1)
    curves_out = np.hstack([curves, div, ratio, label])
    #np.savetxt('curves.csv', curves_out, fmt='%.8f', delimiter=',')
    #np.savetxt('curves_id.csv', ids, fmt='%s', delimiter=',')
    print(curves)

    
    curves_out = np.hstack([curves, div, ratio, label, np.array(ids).reshape(-1,1), np.array(max_curve).reshape(-1,1)])
    df_out = pd.DataFrame(data=curves_out, columns=['Plateau', 'perfast', 'taufast', 'perslow', 'tauslow', 'div_', 'ratio_s', 'div_f', 'ratio_f', 'label_c', 'filename', 'max_dydt'], index=ids)

    cell_type_df = pd.read_csv(in_file_path)
    file_names = cell_type_df['filename'].to_numpy()
    cell_type_df = cell_type_df.set_index('filename')
    #cell_type_label = cell_type_df['cell_label'].to_numpy()
    df_out2 = df_out.join(cell_type_df, on='filename', how='right', lsuffix='_left', rsuffix='_right')
    df3_out = pd.DataFrame.from_dict(sweepwise_data, orient='index')
    
    df_out2.to_csv(out_file_path)

    #write sweepwise data to xlsx with each feature in a separate sheet
    with pd.ExcelWriter(out_file_path.replace('.csv', '_sweepwise.xlsx')) as writer:
        features = ['plateau', 'perfast', 'taufast', 'perslow', 'tauslow', 'div_s', 'ratio_s', 'max_dydt']
        for feature in features:
            df_feature = df3_out.filter(like=feature)
            df_feature.to_excel(writer, sheet_name=feature)

def plot_means(curves, labels, label, div, max_curve):
    means = np.nanmean(curves, axis=0)
    stds = np.nanstd(curves, axis=0)
    plt.figure(figsize=(8,5))
    plt.errorbar(range(len(means)), means, yerr=stds, fmt='o')
    plt.xticks(range(len(means)), labels)
    plt.show()
    means = []
    plt.figure(figsize=(10,10))
    plt.clf()
    for x in np.unique(label).astype(np.int64):
        idx = np.argwhere(label[:,0]==int(x)).astype(np.int32)
        mcur = curves[idx]
        plt.scatter(np.full(len(idx),  x), div[idx], label=label[x])
        means.append(np.nanmean((curves[idx,2]) / (curves[idx,4])))
    plt.legend()
    plt.yscale('log')
    #plt.ylim(0,1)

    print(means)
    means = []
    plt.figure(figsize=(10,10))
    plt.clf()
    for x in np.unique(label).astype(np.int64):
        idx = np.argwhere(label[:,0]==int(x)).astype(np.int32)
        mcur = curves[idx]
        plt.scatter(np.full(len(idx),  x), np.array(max_curve)[idx], label=label[x])
        means.append(np.nanmean((curves[idx,2]) / (curves[idx,4])))
    plt.legend()

    #plt.ylim(0,1)

if __name__ == "__main__":
    process_upwards_growth('/media/smestern/Expansion/PVN_MARM_PROJECT/Clustering March 2023/marm_list.csv', 'marmdata.csv', IC1_file_path='/media/smestern/Expansion/PVN_MARM_PROJECT/Clustering March 2023/Marm Files/**/')
