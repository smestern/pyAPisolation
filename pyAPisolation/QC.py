from loadABF import *
from loadNWB import *
from utils import *
import os
import glob
import pandas as pd
from scipy import stats

def find_zero(realC):
    #expects 1d array
    zero_ind = np.where(realC == 0)[0]
    return zero_ind

def compute_vm_drift(realY, zero_ind):
    sweep_wise_mean = np.mean(realY[:,zero_ind], axis=1)
    mean_drift = np.abs(np.amax(sweep_wise_mean) - np.amin(sweep_wise_mean))
    abs_drift = np.abs(np.amax(realY[:,zero_ind]) - np.amin(realY[:,zero_ind]))
   
    return mean_drift, abs_drift


def compute_rms(realY, zero_ind):
    mean = np.mean(realY[:,zero_ind], axis=1)
    rms = []
    for x in np.arange(mean.shape[0]):
        temp = np.sqrt(np.mean(np.square(realY[x,zero_ind] - mean[x])))
        rms = np.hstack((rms, temp))
    full_mean = np.mean(rms)
    return full_mean, np.amax(rms)

def run_qc(realY, realC):
    zero_ind = find_zero(realC[5,:])
    mean_rms, max_rms = compute_rms(realY, zero_ind)
    mean_drift, max_drift = compute_vm_drift(realY, zero_ind)
    return [mean_rms, max_rms, mean_drift, max_drift]


def main():
    _dir = os.path.dirname(__file__)
    _path = glob.glob(_dir +'//..//data_and_results//HYP_CELL_NWB//Naive//*.nwb')
    full_qc = [0,0,0,0]
    for fp in _path:
        realX, realY, realC = loadNWB(fp)
        temp_qc = run_qc(realY, realC)
        full_qc = np.vstack((full_qc, temp_qc))
    df = pd.DataFrame(data=full_qc[1:,:], columns=['Mean RMS', 'Max RMS', 'Mean Drift', 'Max Drift'], index=_path)
    df.to_csv('qc.csv')
    stats = []
    for col in df.columns.values:
        stats.append(df[col].quantile(0.1))
    qc_stats = pd.DataFrame(data=stats, index=['10 percentile Mean RMS', '10 percentile Max RMS', ' 10 percentile Mean Drift', ' 10 percentile Max Drift'])
    qc_stats.to_csv('qc_stats.csv')

if __name__ == "__main__": 
    main()