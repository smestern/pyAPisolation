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
import pandas as pd

def apfeaturearray(abf):
    
    return 0

def npindofgrt(a, evalp):
    """ Pass through an numpy array and expression, Return indices where eval is true"""
    index = np.nonzero(np.where(a > evalp, 1, 0))
    return index[0] #returns a flattened array of numbers

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
        thresholdavg1 = thresholdavghold[0] #np mean fails if array is 1 value long. So we prevent that by just setting it to the single AP
    else:
        thresholdavg1 = np.nan #return nan if no action potentials are found
    return float(thresholdavg1)


def appreprocess(abf, tag = 'default', save = False, plot = False):
    """ Function takes a given abf file and returns raw and feature data for action potentials across all sweeps. 
        You may wish to use apisolate which returns more fleshed out data
        ______
        abf: An abf file
        tag: if save is turned on, the tag is appeneded to the output files
        save: determines if the raw data is written to a file
        plot: if true, will display a plot of 5 randomly selected aps from the data. Useful for debugging
    """
    
    
    sweepcount = abf.sweepCount
    apcount = 0

    #Build arrays to fill. This has to be pre-created because the size of each ap varies, appending different sized arrays to another makes numpy throw an error. Unused values are truncated later
    aps = np.full((1000, 1000), np.nan)
    peakposDvdt = np.full((1000, 2), np.nan)
    peaknegDvdt = np.full((1000, 2), np.nan)
    peakmV = np.full((1000, 2), np.nan)
    apTime = np.full((1000, 2), np.nan)
    apsweep = np.full((1000, 1), np.nan)
    arthreshold = np.full((1000, 1), np.nan)
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
        for ind, i in enumerate(indexhigher):
               #if i > (aploc):
                    #searches in the next 10ms for the peak    
                    apstrt = (int(i - (abf.dataPointsPerMs * 5)))
                    if apstrt < 0: 
                        apstrt=0
                    apend = int(i + (abf.dataPointsPerMs * 5)) 
                    aploc = np.argmax(abf.sweepY[apstrt:apend]) + apstrt #alternatively aploc = (np.abs(abf.sweepY[apstrt:apend] - thresholdV)).argmin() + apstrt

                    if abf.sweepY[aploc] > -30: #Rejects ap if absolute peak is less than -30mv
                        apstrt = (int(aploc - abf.dataPointsPerMs * 5))
                        thresholdslloc = (np.argmax(slopey[apstrt:aploc]) + apstrt) #Finds the action potential max dvdt
                        nthresholdslloc = (np.argmin(slopey[apstrt:apend]) + apstrt) #Finds the action potential max negative dvdt
                        apstrt = (int(apstrt - abf.dataPointsPerMs * 5))
                        if apstrt < 0:
                            apstrt = 0

                        # Now find the point where DVDT falls below the 5% threshold
                        indexloc = np.nonzero(np.where(slopey[apstrt:thresholdslloc] < thresholdsl, 1, 0))[0]
                        if indexloc.size < 1:
                            idx = apstrt
                        else:
                            indexloc += apstrt
                            idx = indexloc[-1]
                        apstrt = idx
                        ## throw away the ap if the threshold to peak time is more than 2ms
                        if (aploc-idx) > (abf.dataPointsPerMs * 2):
                            break
                        ### Alternatively we can walk through, however above code is much faster
                        ##for y in range(thresholdslloc, 0, -1):
                        #    if slopey[y] < thresholdsl:
                        #        idx = y
                        #        break
                        #    elif y == (thresholdslloc - 800):
                        #        idx = y
                        #        break
                        
                        ## Now we check to ensure the action potentials do not over lap
                        if (ind+1) < (len(indexhigher) - 1):
                            if((indexhigher[ind+1] - indexhigher[ind]) > (abf.dataPointsPerMs * 10)): ##if the next ap is over 10ms away then we simple cap off at 10ms
                                apend = abs(int(aploc + abf.dataPointsPerMs * 10))
                            else:
                                apend = indexhigher[ind+1] #otherwise we cap the end at the next threshold
                                aploc = np.argmax(abf.sweepY[apstrt:apend]) + apstrt
                        else:
                            apend = abs(int(aploc + abf.dataPointsPerMs * 10)) #if this is the last ap in the sweep we cap at 10ms
                        k,  = abf.sweepY.shape
                        if apend > k:
                            apend = int(k) - 1
                        apfull1 = abf.sweepY[apstrt:apend]
                        points = apend - apstrt

                        #Now fill out our arrays
                        peakposDvdt[apcount,0] = slopey[thresholdslloc]
                        peakposDvdt[apcount,1] = abf.sweepX[thresholdslloc]
                        peaknegDvdt[apcount,0] = slopey[nthresholdslloc]
                        peaknegDvdt[apcount,1] = abf.sweepX[nthresholdslloc]
                        peakmV[apcount, 0] = abf.sweepY[aploc]
                        peakmV[apcount, 1] = abf.sweepX[aploc]
                        apTime[apcount, 0] = abf.sweepX[apstrt]
                        apTime[apcount, 1] = abf.sweepX[apend]
                        arthreshold[apcount] = thresholdsl
                        apsweep[apcount] = sweepNumber
                        aps[apcount,:points] = apfull1
                        apcount += 1
        print('Ap count: ' + str(apcount))
    if apcount > 0:
        aps = aps[:apcount,:]
        ## For some Reason truncating these values throws an error, for now they are not truncated
        #peakmV = peakmV[:apcount,:]
        #apTime = apTime[:apcount,:]
        #apsweep = apsweep[:apcount]
        #arthreshold = arthreshold[:apcount]
        #peakposDvdt = peakposDvdt[:apcount] ###Specifically these two throw an error
        #peaknegDvdt = peaknegDvdt[:apcount] ### May be worth it to truncate later
        apsend = np.argwhere(np.invert(np.isnan(aps[:,:])))
        apsend = np.amax(apsend[:,1])
        aps = aps[:,:apsend]
        if plot == True:
            _, l = aps.shape
            test = np.linspace(0, 10, l,endpoint=False)
            for o in range(5):
                j = int(random.uniform(1, apcount - 2))
                plt.plot(test, aps[j,:])
        if save == True:
            np.savetxt('output/' + tag + '.txt', aps, delimiter=",", fmt='%12.5f')
    return aps, abf, peakposDvdt, peaknegDvdt, peakmV, apTime, apsweep, arthreshold, apcount



def apisolate(abf, filter, tag = 'default', saveind = False, savefeat = False, relative = True):
    
    if filter > 0:
       pyabf.filter.gaussian(abf,filter,0)
    aps, abf, peakposDvdt, peaknegDvdt, peakmV, apTime, apsweep, arthreshold, apcount = appreprocess(abf,tag)
    
    _, d = aps.shape
    apoints = np.linspace(0, (d / abf.dataPointsPerMs), d)
    
    
    ## Intialize the rest of the arrays to fill
    dvDtRatio = np.empty((apcount, 1))
    trough = np.empty((apcount, 1))
    slwtrough = np.empty((apcount, 2))
    slwratio = np.empty((apcount, 1))
    fsttrough = np.empty((apcount, 2))
    apheight = np.empty((apcount, 1))
    apwidthloc = np.empty((apcount, 2))
    apfullwidth = np.empty((apcount, 1))
    thresmV = np.empty((apcount, 1))
    apno = np.arange(1, (apcount + 1))
    for i in range(0, apcount - 1):
            abf.setSweep(int(apsweep[i]))
            ### Fill the arrays
            apstrt = int(apTime[i,0] * abf.dataRate)
            aploc = int(peakmV[i,1] * abf.dataRate) - apstrt
            apend = int(apTime[i,1] * abf.dataRate) - apstrt
            thresmV[i] = aps[i,0]
            ttime = (5 * abf.dataPointsPerMs) + aploc
            if ttime > apend:
                ttime = apend
            trough[i] = np.amax(aps[i])
            fsttrough[i, 0] = np.amin(aps[i,aploc:ttime])
            fsttrough[i, 1] = np.argmin(aps[i,aploc:ttime]) + aploc
            if ttime != apend:
                slwtrough[i, 0] = np.amin(aps[i,ttime:apend])
                slwtrough[i, 1] = np.argmin(aps[i,ttime:apend])
            else:
                slwtrough[i] = fsttrough[i]
            apheight[i] = (peakmV[i, 0] - fsttrough[i, 0])
            apwidthloc[i,1] = int((np.argmin(aps[i,aploc:ttime]) - aploc) * 0.5)
            apwidthloc[i,0] = (np.abs(aps[i,:aploc] - (aps[i, int(apwidthloc[i,1])]))).argmin()
            apfullwidth[i] = abf.sweepX[int(apwidthloc[i,1])] - abf.sweepX[int(apwidthloc[i,0])]
            slwratio[i] = ((slwtrough[i, 1] - aploc) / abf.dataRate) / ((apend - aploc) / abf.dataRate)
            aphold = np.array((aps[i], apoints))
            if saveind == True:
                np.savetxt('output/' + str(i) + tag + '.txt', aphold, delimiter=",", fmt='%12.5f')
    
                
    if relative == True:
        peakmV[:,1] = peakmV[:,1] - apTime[:,0]
        peakposDvdt[:,1] = peakposDvdt[:,1] - apTime[:,0]
        peaknegDvdt[:,1] = peaknegDvdt[:,1] - apTime[:,0]
        fsttrough[:, 1] = fsttrough[:, 1] / abf.dataRate
        slwtrough[:, 1] = slwtrough[:, 1] / abf.dataRate
    else:
        fsttrough[:, 1] = (fsttrough[:, 1] / abf.dataRate) + apTime[:,0]
        slwtrough[:, 1] = (slwtrough[:, 1] / abf.dataRate)  + apTime[:,0]
    dvDtRatio[:,0] = peakposDvdt[:apcount, 0] / peaknegDvdt[:apcount, 0]
    

    peakmV = peakmV[:apcount,:]
    apTime = apTime[:apcount,:]
    apsweep = apsweep[:apcount]
    arthreshold = arthreshold[:apcount]
    peakposDvdt = peakposDvdt[:apcount, :] ###Specifically these two throw an error
    peaknegDvdt = peaknegDvdt[:(apcount+1), :] ### May be worth it to truncate later



    
    labels = np.array(['AP Number', 'Sweep', 'Start Time', 'End Time', '5% Threshold', 'mV at Threshold', 'AP Peak (mV)', 'Ap peak (S)', 
                       'AP fast trough (mV)', 'AP fast trough time (S)', 'AP slow trough (mV)', 'AP slow trough time (S)', 'AP slow trough time ratio', 'AP height',
                       'AP Full width (S)', 'AP Upstroke (mV/mS)', 'AP Upstroke time (S)', 'AP downstroke (mV/mS)', 'AP Downstroke time (S)', 'Upstroke / Downstroke Ratio'])
    ardata = np.array([apno, apsweep, apTime[:,0], apTime[:,1], arthreshold, thresmV, peakmV[:,0], peakmV[:,1], fsttrough[:, 0], fsttrough[:, 1], slwtrough[:, 0], slwtrough[:, 1], 
                     slwratio, apheight, apfullwidth, peakposDvdt[:,0], peakposDvdt[:,1], peaknegDvdt[:,0], peaknegDvdt[:,1], dvDtRatio[:,0]])
    tardata = ardata.reshape(-1,1)
    arfrme = pd.DataFrame()
    return 0






