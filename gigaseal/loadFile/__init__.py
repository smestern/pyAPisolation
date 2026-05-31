from .loadNWB import loadNWB, GLOBAL_STIM_NAMES
from .loadABF import loadABF

def loadFile(file_path, return_obj=False, old=False):
    '''
    Loads ABF or NWB files, returns data in the same format. Optionally returns the original object.
    '''
    if file_path.endswith('.abf'):
        return loadABF(file_path, return_obj=return_obj)
    elif file_path.endswith('.nwb'):
        return loadNWB(file_path, return_obj=return_obj, old=old)
    else:
        raise ValueError("Unsupported file type. Only .abf and .nwb files are supported.")