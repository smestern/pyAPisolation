import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from abfderivative import *
import pyabf
from pyabf.tools import *
from pyabf import filter
import os
import pandas as pd
import statistics
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
    for sweepNumber in range(0, sweepcount): 
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
                    #searches in the next 14ms for the peak    
                    apstrt = (int(i - (abf.dataPointsPerMs * 5)))
                    if apstrt < 0: 
                        apstrt=0
                    apend = int(i + (abf.dataPointsPerMs * 3)) 
                    aploc = np.argmax(abf.sweepY[apstrt:apend]) + apstrt #alternatively aploc = (np.abs(abf.sweepY[apstrt:apend] - thresholdV)).argmin() + apstrt

                    if abf.sweepY[aploc] > -30: #Rejects ap if absolute peak is less than -30mv
                        apstrt = (int(aploc - abf.dataPointsPerMs * 5))
                        thresholdslloc = (np.argmax(slopey[apstrt:aploc]) + apstrt) #Finds the action potential max dvdt
                        
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
                        if (ind+1) < (len(indexhigher)):
                            if((indexhigher[ind+1] - indexhigher[ind]) > (abf.dataPointsPerMs * 10)): ##if the next ap is over 10ms away then we simple cap off at 10ms
                                apend = abs(int(aploc + abf.dataPointsPerMs * 10))
                            elif apend > indexhigher[ind+1]:
                                nxtthres = np.nonzero(np.where(slopey[aploc:] >  thresholdsl, 1, 0))[0] + aploc
                                apend = indexhigher[ind+1] #otherwise we cap the end at the next threshold
                                aploc = np.argmax(abf.sweepY[apstrt:apend]) + apstrt #and re-find the peak
                        else:
                            apend = abs(int(aploc + abf.dataPointsPerMs * 10)) #if this is the last ap in the sweep we cap at 10ms
                        k,  = abf.sweepY.shape
                        if apend > k:
                            apend = int(k) - 1
                        apfull1 = abf.sweepY[apstrt:apend]
                        points = apend - apstrt
                        nthresholdslloc = (np.argmin(slopey[aploc:apend]) + aploc) #Finds the action potential max negative dvdt
 
                        #Now fill out our arrays
                        peakposDvdt[apcount,0] = slopey[thresholdslloc]
                        peakposDvdt[apcount,1] = (thresholdslloc - apstrt)
                        peaknegDvdt[apcount,0] = slopey[nthresholdslloc]
                        peaknegDvdt[apcount,1] = (nthresholdslloc - apstrt)
                        peakmV[apcount, 0] = abf.sweepY[aploc]
                        peakmV[apcount, 1] = (aploc - apstrt)
                        apTime[apcount, 0] = apstrt
                        apTime[apcount, 1] = points
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



def apisolate(abf, filter, tag = 'default', saveind = False, savefeat = False, plot = 0):
    """ Function takes a given abf file and returns raw and feature data for action potentials across all sweeps. 
        The data is returned in a feature complete way. Saving requires the creation of an '/output' folder
        ---Takes---
        abf: An abf file
        Filter: if nonzero applies a gaussian filter to the data (useful if noisy)
        tag: if save is turned on, the tag is appeneded to the output files
        saveind: Saves the individual aps as raw traces
        savefeat: Saves the feature array of all action potentials to a file
        relative: if true, calculated features are based on thier time post threshold, and not in respect to the start of the sweep
        plot: Int, plots a randomly selected (int)number of aps from the abf, with the features highlighted. Call matplotlib.plot.show() to see
        ---Returns---
        aps: the raw current traces in a numpy array
        tarframe: the feature array in a pandas data frame (ONLY if savefeat = true, otherwise returns 0)
        abf: the original abf file passed to the function
    """
    relative = True

    if filter > 0:
       pyabf.filter.gaussian(abf,filter,0)
       np.nan_to_num(abf.data, nan=-9999, copy=False)
    aps, abf, peakposDvdt, peaknegDvdt, peakmV, apTime, apsweep, arthreshold, apcount = appreprocess(abf,tag,False)
    
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
    isi = np.empty(apcount)
    apno = np.arange(0, (apcount + 1))
    for i in range(0, apcount):
            abf.setSweep(int(apsweep[i]))
            ### Fill the arrays if we need to
            apstrt = int(apTime[i,0])
            aploc = int(peakmV[i,1])
            apend = int(apTime[i,1])
            thresmV[i] = aps[i,0]
            ttime = (5 * abf.dataPointsPerMs) + aploc
            if ttime > apend:
                    ttime = apend
            trough[i] = np.amax(aps[i])
            fsttrough[i, 0] = np.amin(aps[i,aploc:ttime])
            fsttrough[i, 1] = np.argmin(aps[i,aploc:ttime]) + aploc
            if ttime != apend: 
                   slwtrough[i, 0] = np.amin(aps[i,ttime:apend])
                   slwtrough[i, 1] = np.argmin(aps[i,ttime:apend]) + ttime
            else:
                    slwtrough[i] = fsttrough[i]
            apheight[i] = (peakmV[i, 0] - fsttrough[i, 0])
            if i != (apcount-1) and apsweep[i+1] == apsweep[i]:
                    isi[i] = abs(apTime[i, 0] - apTime[i+1, 0]) / abf.dataRate
            else:
                    isi[i] = abs((apTime[i, 0] / abf.dataRate) - abf.sweepX[-1])
            aphalfheight = statistics.median([peakmV[i, 0], fsttrough[i, 0]])
            #apwidthloc[i,1] = int((np.argmin(aps[i,aploc:ttime]) + aploc) * 0.5)
            apwidthloc[i,1] = (np.abs(aps[i, aploc:ttime] - aphalfheight)).argmin() + aploc
            apwidthloc[i,0] = (np.abs(aps[i,:aploc] - (aps[i, int(apwidthloc[i,1])]))).argmin()
            apfullwidth[i] = (apwidthloc[i,1] - apwidthloc[i,0]) / abf.dataRate
            slwratio[i] = ((slwtrough[i, 1] - aploc) / abf.dataRate) / ((apend - aploc) / abf.dataRate)

            
  
    peakmV[:,1] = peakmV[:,1] / abf.dataRate
    peakposDvdt[:,1] = peakposDvdt[:,1] / abf.dataRate
    peaknegDvdt[:,1] = peaknegDvdt[:,1] / abf.dataRate
    fsttrough[:, 1] = fsttrough[:, 1] / abf.dataRate
    slwtrough[:, 1] = slwtrough[:, 1] / abf.dataRate
    dvDtRatio[:,0] = peakposDvdt[:apcount, 0] / peaknegDvdt[:apcount, 0]
    peakmV = peakmV[:apcount,:]
    apTime = apTime[:apcount,:] / abf.dataRate
    apsweep = apsweep[:apcount]
    arthreshold = arthreshold[:apcount]
    peakposDvdt = peakposDvdt[:apcount, :] ###Specifically these two throw an error
    peaknegDvdt = peaknegDvdt[:(apcount+1), :] ### this fixes it, im not sure why

    if plot > 0 and apcount > 0:
            _, l = aps.shape
            xdata = np.linspace(0, 10, l,endpoint=True)
            for o in range(plot):
                j = int(random.uniform(1, apcount - 2))
                plt.plot(xdata, aps[j,:])
                q = int(peaknegDvdt[j,1]  * abf.dataRate)
                plt.plot(xdata[q], aps[j,q], 'rx', label='Peak Neg dVdT')
                q = int(peakposDvdt[j,1]  * abf.dataRate)
                plt.plot(xdata[q], aps[j,q], 'bx', label='Peak Pos dVdT')
                q = int(peakmV[j,1] * abf.dataRate)
                plt.plot(xdata[q], aps[j,q], 'gx', label='Peak mV')
                q = int(slwtrough[j, 1] * abf.dataRate)
                plt.plot(xdata[q], aps[j,q], 'r>', label='Slow Trough')
                q = int(fsttrough[j, 1] * abf.dataRate)
                plt.plot(xdata[q], aps[j,q], 'b>', label='Fst Trough')
                q = int(apwidthloc[i,0])
                q2 = int(apwidthloc[i,1])
                
                plt.plot(xdata[q], aps[j,q2], 'yx')
                plt.plot(xdata[q2], aps[j,q2], 'yx')
                plt.plot(xdata[q:q2], np.full(((q2-q)), aps[j,q2]), 'y-', solid_capstyle='round', label='Full Width')
                
    
    ## If saving the feature array we need to construct the labels
    labels = np.array(['AP Number', 'Sweep', 'Start Time', 'End Time', 'ISI', '5% Threshold', 'mV at Threshold', 'AP Peak (mV)', 'Ap peak (S)', 
                       'AP fast trough (mV)', 'AP fast trough time (S)', 'AP slow trough (mV)', 'AP slow trough time (S)', 'AP slow trough time ratio', 'AP height',
                       'AP Full width (S)', 'AP Upstroke (mV/mS)', 'AP Upstroke time (S)', 'AP downstroke (mV/mS)', 'AP Downstroke time (S)', 'Upstroke / Downstroke Ratio'])
    ## We could put it in a numpy array, but arrays of different types slow down the code...
    #ardata = np.vstack((apno[:-1], apsweep[:,0], apTime[:,0], apTime[:,1], isi, arthreshold[:,0], thresmV[:,0], peakmV[:,0], peakmV[:,1], fsttrough[:, 0], fsttrough[:, 1], slwtrough[:, 0], slwtrough[:, 1], 
    #               slwratio[:,0], apheight[:,0], apfullwidth[:,0], peakposDvdt[:,0], peakposDvdt[:,1], peaknegDvdt[:-1,0], peaknegDvdt[:-1,1], dvDtRatio[:,0]))
    ### Or we dump it into a panda dataframe. Faster / handles better than a numpy array
    arfrme = pd.DataFrame([apsweep[:,0], apTime[:,0], apTime[:,1], isi, arthreshold[:,0], thresmV[:,0], peakmV[:,0], peakmV[:,1], fsttrough[:, 0], fsttrough[:, 1], slwtrough[:, 0], slwtrough[:, 1], 
                     slwratio[:,0], apheight[:,0], apfullwidth[:,0], peakposDvdt[:,0], peakposDvdt[:,1], peaknegDvdt[:,0], peaknegDvdt[:,1], dvDtRatio[:,0]],
                          index=labels[1:],
                          columns=apno[:])
    tarfrme = arfrme.T[:apcount] ##Transpose for organization reasons
    
    ##Check one more time for duplicates
    height = np.nonzero(np.where(isi == 0, 1, 0)) ##finding only indicies where ISI == 0
    for z in height[0]:
        if z != (apno[-1] - 1):
            if apsweep[z,0] == apsweep[z+1,0]:
                tarfrme = tarfrme[tarfrme.index != z]
                aps = np.delete(aps, z, 0)
                #ardata = np.delete(ardata, z, 1)
    if savefeat == True:
        tarfrme.to_csv('output/feat' + tag + '.csv')
        print('feat' + tag + '.csv saved')
    ## Save raw traces if we need to
    if saveind == True:
        for m in range(0, apcount - 1):
                aphold = np.array((aps[m], apoints))
                np.savetxt('output/' + str(m) + tag + '.csv', aphold, delimiter=",", fmt='%12.5f')
    return aps, tarfrme, abf






