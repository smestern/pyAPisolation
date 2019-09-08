#!/usr/bin/evn python

import numpy as np
from numpy import genfromtxt
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from abfderivative import *
import pyabf
from pyabf import filter
from matplotlib import cm
import tkinter as tk
from tkinter import filedialog
import os

root = tk.Tk()
root.withdraw()


def apfeaturearray(abf):
    
    return 0

def npconsec(a):
    index = np.nonzero(np.where(np.ediff1d(a) > 1, 1, 0))
    return index[0]

def thresholdavg(abf, sweep, thresdvdt = 20):
    abf.setSweep(sweep)
    apend = 0
    aploc = 0
    slopex, slopey = derivative(abf,sweep,1)
    thresholdavghold = np.empty((1))
    np.nan_to_num(abf.sweepY, nan=-9999, copy=False)
    np.nan_to_num(slopey, nan=0, copy=False)
    indexhigher = np.nonzero(np.where(slopey > thresdvdt, 1, 0)) #Returns indices only where the slope is greater than the threshold
    indexhigher = indexhigher[0] #flattens
    for j in indexhigher: 
        if j > aploc:
            k = slopey[j]
            apend = int(j + (abf.dataPointsPerMs * 5)) #searches in the next 10ms for the peak
            apstrt = int(j - (abf.dataPointsPerMs * 2))
            if apstrt < 0: 
                        apstrt=0
            aploc = np.argmax(abf.sweepY[apstrt:apend]) + apstrt
            if abf.sweepY[aploc] > -30: #Rejects ap if absolute peak is less than -30mv
                maxdvdt = np.amax(slopey[apstrt:aploc])
                thresholdavghold = np.append(thresholdavghold, maxdvdt)
    thresholdavg = np.mean(thresholdavghold[1:]) 
    return thresholdavg

def appreprocess2(abf, tag = 'default', save = False, plot = False):
    apf = False
    apstrt = 0
    apend = 0
    fileno, void = tag.split('-')
    sweepcount = 1
    apcount = 0
    aps = np.full((5000, 5000), np.nan)
    aplochold = 0
    if abf.sweepCount > 1:
        sweepcount = (abf.sweepCount - 1)
    for sweepNumber in range(sweepcount):
        print(sweepNumber)
        abf.setSweep(sweepNumber)
        aploc = 0
        idx = 0
        thresholdV = np.amax(abf.sweepY)
        thresholdsl = (thresholdavg(abf,sweepNumber) * 0.05)
        print(thresholdsl)
        apf = False
        slopex, slopey = derivative(abf,sweepNumber,1)
        np.nan_to_num(abf.sweepY, nan=-9999, copy=False)
        np.nan_to_num(slopey, nan=0, copy=False)
        indexhigher = np.nonzero(np.where(slopey > thresholdsl, 1, 0)) #Returns indices only where the slope is greater than the threshold
        indexhigher = indexhigher[0] #flattens
        p = npconsec(indexhigher)
        print(p)
        for i in indexhigher:
               if i > (aploc + (abf.dataPointsPerMs * 2)):
                    apstrt = (int(i - (abf.dataPointsPerMs * 5)))
                    if apstrt < 0: 
                        apstrt=0
                    apend = int(i + (abf.dataPointsPerMs * 20)) #searches in the next 20ms for the peak
                    #aploc = (np.abs(abf.sweepY[apstrt:apend] - thresholdV)).argmin() + apstrt
                    aploc = np.argmax(abf.sweepY[apstrt:apend]) + apstrt
                    if abf.sweepY[aploc] > -30: #Rejects ap if absolute peak is less than -30mv
                        apstrt = (int(aploc - abf.dataPointsPerMs * 5))
                        thresholdslloc = (np.argmax(slopey[apstrt:aploc]) + apstrt)
                        apstrt = (int(apstrt - abf.dataPointsPerMs * 5))
                        if apstrt < 0:
                            apstrt = 0
                        for y in range(thresholdslloc, 0, -1):
                            if slopey[y] < thresholdsl:
                                idx = y
                                break
                            elif y == (thresholdslloc-5000):
                                idx = y
                                break
                        #idx = (np.abs(slopey[apstrt:thresholdslloc] - thresholdsl)).argmin()
                        if idx < 1: 
                            print(abf.sweepY[idx])
                            print(thresholdsl)
                        apstrt = idx
                        apend = abs(int(aploc + abf.dataPointsPerMs * 20))
                        k,  = abf.sweepY.shape
                        if apend > k:
                            apend = int(k)
                        apfull1 = abf.sweepY[apstrt:apend]
                        points = apend - apstrt
                        #plt.plot(np.linspace(0,points,points), apfull1)
                        aps[apcount,:points] = apfull1
                        apcount += 1
    if apcount > 0:
        aps = aps[1:apcount,:]
        apsend = np.argwhere(np.invert(np.isnan(aps[:,:])))
        apsend = np.amax(apsend[:,1])
        aps = aps[:,:apsend]
        if plot == True:
            void, l = aps.shape
            test = np.linspace(0, l, num=l,endpoint=False)
            for o in range(5):
                j = int(random.uniform(1, apcount - 1))
                if abf.dataRate > 10000:
                    test = test[:int(l/2)]
                    #plt.plot(test, aps[j,::2])
                else:
                    plt.plot(test, aps[j,:])
    
        if save == True:
            np.savetxt('output/' + tag + '.txt', aps, delimiter=",", fmt='%12.5f')
    return aps, abf

def apisolate(abf, threshold, filter, tag = 'default', save = False):
    aps, thresholdavg, abf = appreprocess(abf,tag,save)
    



    return 0






