#!/usr/bin/evn python

import numpy as np
from numpy import genfromtxt
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from abfderivative import *
import pyabf
from pyabf.tools import *
from pyabf import filter
from matplotlib import cm
import os


def apfeaturearray(abf):
    
    return 0

def npindofgrt(a, evalp):
    """ Pass through an numpy array and expression, Return indices where eval is true"""
    index = np.nonzero(np.where(a > evalp, 1, 0))
    return index[0]

def thresholdavg(abf, sweep, thresdvdt = 20):
    """ Given an ABF file and a sweep, this function returns the avg max DVdT of action potentials in a sweep.
       ABF: a pyabf object
       Sweep: The sweep number for analysis
       ThresDVDT: Optional, the dVdT cut off. Defaults to the allen institute's 20mv/ms 
    """
    
    abf.setSweep(sweep)

    #Define our variables for error purposes
    apend = 0
    aploc = 0
    thresholdavg1 = np.nan
    thresholdavghold = np.empty((1))

    slopex, slopey = derivative(abf,sweep) #Pass through the derivative function
    indexhigher = pyabf.tools.ap.ap_points_currentSweep(abf, thresdvdt) #Returns indices only where the slope is greater than the threshold. Using the built in functions for now. Otherwise index = np.nonzero(np.where(slopey > Threshold, 1, 0)) would work
    for j in indexhigher: #iterates through the known threshold values 
            k = slopey[j]
            #searches in the next 10ms for the peak
            apend = int(j + (abf.dataPointsPerMs * 5)) 
            apstrt = int(j - (abf.dataPointsPerMs * 5))
            if apstrt < 0: 
                        apstrt=0

            aploc = np.argmax(abf.sweepY[apstrt:apend]) + apstrt #Finds the peak mV of within 10ms of ap
            if abf.sweepY[aploc] > -30: #Rejects ap if absolute peak is less than -30mv
                maxdvdt = np.amax(slopey[apstrt:aploc])
                thresholdavghold = np.append(thresholdavghold, maxdvdt) #otherwise adds the value to our array
    thresholdavghold = thresholdavghold[1:] #truncates the intial value which was meaningless
    l, = thresholdavghold.shape 
    if l > 1:
        thresholdavg1 = np.mean(thresholdavghold)
    elif l == 1:
        thresholdavg1 = thresholdavghold[0] #np mean fails if array is 1 long. So we prevent that by just setting it to the single AP
    else:
        thresholdavg1 = np.nan #return nan if no action potentials are found
    return float(thresholdavg1)


def appreprocess(abf, tag = 'default', save = False, plot = False):
    sweepcount = abf.sweepCount
    apcount = 0

    #Build arrays to fill. This has to be pre-created as size of ap varies. Unused values are truncated later
    aps = np.full((1000, 1000), np.nan)
    peakposDvdt = np.full((1000, 2), np.nan)
    peaknegDvdt = np.full((1000, 2), np.nan)
    peakmV = np.full((1000, 2), np.nan)
    apTime = np.full((1000, 2), np.nan)

    
    #If there is more than one sweep, we need to ensure we dont iterate out of range
    if abf.sweepCount > 1:
        sweepcount = (abf.sweepCount - 1)

    #Now we walk through the sweeps looking for action potentials
    for sweepNumber in range(sweepcount): 
        print(sweepNumber)
        abf.setSweep(sweepNumber)
        aploc = 0
        idx = 0
        thresholdV = np.amax(abf.sweepY)
        thresholdsl = (thresholdavg(abf, sweepNumber) * 0.05)
        print('%5 threhold avg: ' + str(thresholdsl))
        slopex, slopey = derivative(abf,sweepNumber)
        np.nan_to_num(abf.sweepY, nan=-9999, copy=False)
        np.nan_to_num(slopey, nan=0, copy=False)
        indexhigher = pyabf.tools.ap.ap_points_currentSweep(abf)
        print('Ap count: ' + str(apcount))
        ind = 0
        for i in indexhigher:
               #if i > (aploc):
                    apstrt = (int(i - (abf.dataPointsPerMs * 5)))
                    if apstrt < 0: 
                        apstrt=0
                    apend = int(i + (abf.dataPointsPerMs * 5)) #searches in the next 10ms for the peak
                    #aploc = (np.abs(abf.sweepY[apstrt:apend] - thresholdV)).argmin() + apstrt
                    aploc = np.argmax(abf.sweepY[apstrt:apend]) + apstrt
                    if abf.sweepY[aploc] > -30: #Rejects ap if absolute peak is less than -30mv
                        apstrt = (int(aploc - abf.dataPointsPerMs * 5))
                        thresholdslloc = (np.argmax(slopey[apstrt:aploc]) + apstrt) #Finds the action potential max dvdt
                        apstrt = (int(apstrt - abf.dataPointsPerMs * 5))
                        if apstrt < 0:
                            apstrt = 0
                        for y in range(thresholdslloc, 0, -1):
                            if slopey[y] < thresholdsl:
                                idx = y
                                break
                            elif y == (thresholdslloc - 800):
                                idx = y
                                break
                        okoko = 0
                        if idx < 1: 
                            print(abf.sweepY[idx])
                            print(i)
                            print(thresholdsl)
                            
                        apstrt = idx
                        if (ind+1) < (len(indexhigher) - 1):
                            if((indexhigher[ind+1] - indexhigher[ind]) > (abf.dataPointsPerMs * 10)):
                                apend = abs(int(aploc + abf.dataPointsPerMs * 10))
                            else:
                                apend = indexhigher[ind+1]
                        else:
                            apend = abs(int(aploc + abf.dataPointsPerMs * 10))
                        k,  = abf.sweepY.shape
                        if apend > k:
                            apend = int(k)
                        apfull1 = abf.sweepY[apstrt:apend]
                        points = apend - apstrt

                        #Now fill out our arrays
                        peakposDvdt[apcount,0] = slopey[thresholdslloc]
                        peakposDvdt[apcount,0] = abf.sweepX[thresholdslloc]
                        aps[apcount,:points] = apfull1
                        apcount += 1
                        ind += 1
    if apcount > 0:
        print(apcount)
        #aps = aps[1:apcount,:]
        apsend = np.argwhere(np.invert(np.isnan(aps[:,:])))
        apsend = np.amax(apsend[:,1])
        aps = aps[:,:apsend]
        if plot == True:
            void, l = aps.shape
            test = np.linspace(0, 10, l,endpoint=False)
            for o in range(80):
                j = int(random.uniform(1, apcount - 2))
                plt.plot(test, aps[j,:])
        if save == True:
            np.savetxt('output/' + tag + '.txt', aps, delimiter=",", fmt='%12.5f')
    return aps, abf


def appreprocess2(abf, tag = 'default', save = False, plot = False):
    #### For now this is uneeded. New Method found in appreprocess is better

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
        print(p.shape)
        print(apcount)
        for i in indexhigher:
               if i > (aploc):
                    apstrt = (int(i - (abf.dataPointsPerMs * 5)))
                    if apstrt < 0: 
                        apstrt=0
                    apend = int(i + (abf.dataPointsPerMs * 5)) #searches in the next 20ms for the peak
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
                            elif y == (thresholdslloc - 800):
                                idx = y
                                break
                        #idx = (np.abs(slopey[apstrt:thresholdslloc] - thresholdsl)).argmin()
                        if idx < 1: 
                            print(abf.sweepY[idx])
                            print(thresholdsl)
                            print(i)
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






