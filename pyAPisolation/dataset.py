import pandas as pd
import numpy as np
import logging
from ipfx.sweep import Sweep, SweepSet
from .loadFile import loadFile
import os
try:
    from ipfx.dataset.ephys_data_set import EphysDataSet
    from ipfx.stimulus import StimulusOntology
    from ipfx.dataset.ephys_nwb_data import EphysNWBData, get_finite_or_none
    from ipfx.dataset.hbg_nwb_data import HBGNWBData
except ImportError:
    print(f"Error importing from ipfx.dataset.ephys_data_set, ipfx.stimulus, ipfx.dataset.ephys_nwb_data, ipfx.dataset.hbg_nwb_data")
    print(f"Likely not an issue, this message is for Sam")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class cellData(object):
    """
    A celldata object, that stores the data for a single cell, and allows for easy access to the data. This allows the data to be easily passed around and manipulated
    The data is stored in the dataX, dataY, and dataC arrays, which are the stimulus, response, and command arrays respectively.
    The object follows pyABF conventions, where the dataX array is the time array, the dataY array is the response array, and the dataC array is the command array.
    One of file, or (dataX, dataY, dataC) must be provided. 
    If file is provided, the data will be loaded from the file, otherwise the data will be loaded from the dataX, dataY, and dataC arrays
    Takes:
        file (optional): str, path to the file to load the data from
        dataX (optional): np.array, the time data
        dataY (optional): np.array, the response data
        dataC (optional): np.array, the command data
        name (optional): str, the name of the data  (default: generated from the dataY array)
        clampMode (optional): str, the clamp mode of the data (default: None)
        stimUnits (optional): str, the units of the stimulus data (default: 'pA')
        respUnits (optional): str, the units of the response data (default: 'mV')
    returns:
        cellData object
    """

    def __init__(self, file=None, dataX=None, dataY=None, dataC=None, name=None, protocolList=None, clampMode=None, stimUnits='pA', respUnits='mV'):
        logger.info(f"Creating cellData object")
        # if the file is not none, then we are loading from a file
        self.protocolList = protocolList
        self.protocol = None
        if file is not None:
            logger.info(f"Loading data from file: {file}")
            #if protocol list is provided, set the protocol
            self.dataX, self.dataY, self.dataC, self._file_obj = loadFile(file, return_obj=True)
            self.data = [self.dataX, self.dataY, self.dataC]
            self.file = file
            self.fileName = file.split('/')[-1]
            self.filePath = os.path.abspath(file)
            self.name = '.'.join(os.path.basename(file).split('.')[:-1])
            self._load_from_file()
            self.protocol = self._file_obj.protocol
        else:
            logger.info(f"Loading data from arrays")
            self.data = None
            self.file = None
            self.fileName = ''
            self.filePath = ''
            if name is not None:
                self.name = name
            else:
                logger.info(f"Generating name from data")
                # create a unique name by hashing the data
                self.name = "unamed_" + str(hash(dataY[0].tostring()))

            self.dataX = dataX
            self.dataY = dataY
            self.dataC = dataC
            self.data = [self.dataX, self.dataY, self.dataC]
            self.clampMode = clampMode
            self.stimUnits = stimUnits
            self.respUnits = respUnits

        #default values
        self.setSweep(0)


    def _load_from_file(self):
        self.dataX = self.data[0]
        self.dataY = self.data[1]
        self.dataC = self.data[2]

    def setProtocol(self, protocol):
        """ Some files / datasets may have multiple protocols, this allows the user to set the protocol for the data.
        This is useful for when the data is loaded from a file, and the protocol is not known
        """
        self.protocol = protocol

    #pyABF.abf like properties
    def setSweep(self, sweep):
        self.sweep = sweep
        # index into the dataX, dataY, and dataC arrays to get the sweep data
        self.sweepX = self.dataX[sweep]
        self.sweepY = self.dataY[sweep]
        self.sweepC = self.dataC[sweep]
        
    @property
    def sweepList(self):
        return list(range(len(self.dataX)))

    @property
    def sweepCount(self):
        return len(self.dataX)
    
    @property
    def sweepNumber(self):
        return self.sweep
    
    @property
    def sweepLengthSec(self):
        return self.sweepX[-1]

    def __str__(self):
        return f"cellData object: {self.name}, loaded from {self.file}"
    
    def __repr__(self):
        return f"cellData object: {self.name}, loaded from {self.file}"
    
    def __getitem__(self, key):
        return self.data[key]
    
    
