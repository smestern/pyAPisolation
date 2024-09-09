import numpy as np
from .loadABF import loadABF
try:
    import h5py
    ##Does not import when using python-matlab interface on windows machines
except:
    print("h5py import fail")
import pandas as pd

def loadFile(file_path, return_obj=False, old=False):
    """Loads the nwb object and returns three arrays dataX, dataY, dataC and optionally the object.
    same input / output as loadABF for easy pipeline inclusion

    Args:
        file_path (str): [description]
        return_obj (bool, optional): return the NWB object to access various properites. Defaults to False.
        old (bool, optional): use the old indexing method, uneeded in most cases. Defaults to False.

    Returns:
        dataX: time (should be seconds)
        dataY: voltage (should be mV)
        dataC: current (should be pA)
        dt: time step (should be seconds)
    """    
    if file_path.endswith(".nwb"):
        return loadNWB(file_path, return_obj, old)
    elif file_path.endswith(".abf"):
        return loadABF(file_path, return_obj)
    else:
        raise Exception("File type not supported")



def loadNWB(file_path, return_obj=False, old=False, load_into_mem=True):
    """Loads the nwb object and returns three arrays dataX, dataY, dataC and optionally the object.
    same input / output as loadABF for easy pipeline inclusion

    Args:
        file_path (str): [description]
        return_obj (bool, optional): return the NWB object to access various properites. Defaults to False.
        old (bool, optional): use the old indexing method, uneeded in most cases. Defaults to False.
        load_into_mem (bool, optional): load the data into memory. Defaults to True.        load_into_mem (bool, optional): load the data into memory. Defaults to True.

    Returns:
        dataX: time (should be seconds)
        dataY: voltage (should be mV)
        dataC: current (should be pA)
        dt: time step (should be seconds)
        obj: the nwb object (optional)
    """    
   
    if old:
        nwb = old_nwbFile(file_path)
    else:
        nwb = nwbFile(file_path, load_into_mem=load_into_mem)
    
    fs_dict = nwb.rate # sampling rate info
    fs = fs_dict["rate"] # assumes units of Hz
    dt = np.reciprocal(fs) # seconds
    
    if isinstance(nwb.dataX, np.ndarray)==False and load_into_mem==True:
        dataX = np.asarray(nwb.dataX, dtype=np.dtype('O')) ##Assumes if they are still lists its due to uneven size
        dataY = np.asarray(nwb.dataY, dtype=np.dtype('O')) #Casts them as numpy object types to deal with this
        dataC = np.asarray(nwb.dataC, dtype=np.dtype('O'))
    elif load_into_mem==True:
        dataX = nwb.dataX #If they are numpy arrays just pass them
        dataY = nwb.dataY
        dataC = nwb.dataC
    else:
        ##If not loading into memory just pass the lists
        dataX = []
        dataY = []
        dataC = []

    if return_obj == True:
        return dataX, dataY, dataC, nwb
    else:
        return dataX, dataY, dataC, 

    ##Final return incase if statement fails somehow
    return dataX, dataY, dataC



# A simple class to load the nwb data quick and easy
##Call like nwb = nwbfile('test.nwb')
##Sweep data is then located at nwb.dataX, nwb.dataY, nwb.dataC (for stim)
class nwbFile(object):

    def __init__(self, file_path, load_into_mem=True):
        with h5py.File(file_path,  "r") as f:
            ##Load some general properities
            acq_keys = list(f['acquisition'].keys())
            stim_keys = list(f['stimulus']['presentation'].keys())
            sweeps = zip(acq_keys, stim_keys)##Sweeps are stored as keys
            
            #self.temp = f['general']['Temperature'][()]
            ## Find the index's with long square
            index_to_use = []
            for key_resp, key_stim in sweeps: 
                sweep_dict = dict(f['acquisition'][key_resp].attrs.items())
                if check_stimulus(sweep_dict, key_resp):
                    index_to_use.append((key_resp, key_stim)) 
            self.sweepCount = len(index_to_use)
            if len(index_to_use)==0:
                #set rate etc to nan
                self.rate = {'rate':np.nan}
                self.sweepYVars = np.nan
                self.sweepCVars = np.nan
            else:
                self.rate = dict(f['acquisition'][index_to_use[0][0]]['starting_time'].attrs.items())
                self.sweepYVars = dict(f['acquisition'][index_to_use[0][0]]['data'].attrs.items())
                self.sweepCVars = dict(f['stimulus']['presentation'][stim_keys[-1]]['data'].attrs.items())
           
            dataY = []
            dataX = []
            dataC = []
            self.sweepMetadata = []
            for sweep_resp, sweep_stim in index_to_use:
                ##Load the response and stim
                data_space_s = 1/(dict(f['acquisition'][sweep_resp]['starting_time'].attrs.items())['rate'])
                try:
                    bias_current = f['acquisition'][sweep_resp]['bias_current'][()]
                    if np.isnan(bias_current):
                        #continue
                        bias_current = 0
                except:
                    bias_current = 0
                
                if load_into_mem==True:
                    temp_dataY = np.asarray(f['acquisition'][sweep_resp]['data'][()]) * dict(f['acquisition'][sweep_resp]['data'].attrs.items())['conversion'] 
                    temp_dataX = np.cumsum(np.hstack((0, np.full(temp_dataY.shape[0]-1,data_space_s))))
                    temp_dataC = np.asarray(f['stimulus']['presentation'][sweep_stim]['data'][()]) * dict(f['stimulus']['presentation'][sweep_stim]['data'].attrs.items())['conversion']
                    dataY.append(temp_dataY)
                    dataX.append(temp_dataX)
                    dataC.append(temp_dataC)
                else:
                    dataY.append(f['acquisition'][sweep_resp]['data'])
                    dataX.append(f['acquisition'][sweep_resp]['starting_time'])
                    dataC.append(f['stimulus']['presentation'][sweep_stim]['data'])
                sweep_dict_resp = dict(f['acquisition'][sweep_resp].attrs.items())
                sweep_dict_resp.update(dict(f['acquisition'][sweep_resp]['data'].attrs.items()))
                sweep_dict_stim = dict(f['stimulus']['presentation'][sweep_stim].attrs.items())
                sweep_dict_stim.update(dict(f['stimulus']['presentation'][sweep_stim]['data'].attrs.items()))
                self.sweepMetadata.append(dict(resp_dict = sweep_dict_resp, stim_dict=sweep_dict_stim))
            try:
                ##Try to vstack assuming all sweeps are same length
                self.dataX = np.vstack(dataX)
                self.dataC = np.vstack(dataC)
                self.dataY = np.vstack(dataY)
            except:
                #Just leave as lists
                self.dataX = dataX
                self.dataC = dataC
                self.dataY = dataY
        return

class stim_names:
    stim_inc = ['long', '1000']
    stim_exc = ['rheo', 'Rf50_']
    stim_type = ['']
    def __init__(self):
        self.stim_inc = stim_names.stim_inc
        self.stim_exc = stim_names.stim_exc
        return

GLOBAL_STIM_NAMES = stim_names()

def check_stimulus(sweep_dict, name):
    desc_check = np.any([check_stimulus_desc(sweep_dict['description']), check_stimulus_desc(sweep_dict['stimulus_description'])])
    type_check = check_stimulus_type(sweep_dict['neurodata_type']) or check_stimulus_type(name)
    return np.logical_and(desc_check, type_check)

def check_stimulus_type(sweep_type):
    try:
        sweep_type_str = sweep_type.decode()
    except:
        sweep_type_str = sweep_type
    return np.any([x.upper() in sweep_type_str.upper() for x in GLOBAL_STIM_NAMES.stim_type])

def check_stimulus_desc(stim_desc):
    try:
        stim_desc_str = stim_desc.decode() #sometimes its encoded... sometimes its not
    except:
        stim_desc_str = stim_desc
    #print(stim_desc_str)
    include_s = np.any([x.upper() in stim_desc_str.upper() for x in GLOBAL_STIM_NAMES.stim_inc])
    exclude_s = np.invert(np.any([x.upper() in stim_desc_str.upper() for x in GLOBAL_STIM_NAMES.stim_exc]))
    return np.logical_and(include_s, exclude_s)




class old_nwbFile(object):
    """
    A simple class to load the nwb data quick and easy. This handles older nwb files that do not have the same structure as the newer ones.
    """
    def __init__(self, file_path):
        with h5py.File(file_path,  "r") as f:
            ##Load some general properities
            sweeps = list(f['acquisition'].keys()) ##Sweeps are stored as keys
            self.sweepCount = len(sweeps)
            self.rate = dict(f['acquisition'][sweeps[0]]['starting_time'].attrs.items())
            self.sweepYVars = dict(f['acquisition'][sweeps[0]]['data'].attrs.items())
            self.sweepCVars = dict(f['stimulus']['presentation'][sweeps[0]]['data'].attrs.items())
            ##Load the response and stim
            data_space_s = 1/self.rate['rate']
            dataY = []
            dataX = []
            dataC = []
            for sweep in sweeps:
                ##Load the response and stim
                data_space_s = 1/(dict(f['acquisition'][sweeps[0]]['starting_time'].attrs.items())['rate'])
                temp_dataY = np.asarray(f['acquisition'][sweep]['data'][()])
                temp_dataX = np.cumsum(np.hstack((0, np.full(temp_dataY.shape[0]-1,data_space_s))))
                temp_dataC = np.asarray(f['stimulus']['presentation'][sweep]['data'][()])
                dataY.append(temp_dataY)
                dataX.append(temp_dataX)
                dataC.append(temp_dataC)
            try:
                ##Try to vstack assuming all sweeps are same length
                self.dataX = np.vstack(dataX)
                self.dataC = np.vstack(dataC)
                self.dataY = np.vstack(dataY)
            except:
                #Just leave as lists
                self.dataX = dataX
                self.dataC = dataC
                self.dataY = dataY
        return