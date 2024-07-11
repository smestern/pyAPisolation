import numpy as np
import pandas as pd
import os
import glob
import sys
from scipy.signal import resample, decimate
import matplotlib.pyplot as plt
import anndata as ad
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class experimentalStructure:
    """
    A class to represent the experimental structure of a time series database. Each entry represents a protocol,
    and each protcol has a set of flags, (e.g. time, cell type, etc.).
    One flag is the primairy flag, which is used to flag the protocool the represents the primary time series (for computing features etc).
    """
    def __init__(self):
        """
        Constructor for the experimentalStructure class
        :param path: Path to the database
        :param exp: Experimental structure of the database
        """
        self.protocols = []
        self.primary = None
        self.flags = {}

    def addProtocol(self, name, flags):
        """
        Add a protocol to the experimental structure
        :param name: Name of the protocol
        :param flags: Flags of the protocol
        """
        self.protocols.append(name)
        for flag in flags:
            if flag not in self.flags:
                self.flags[flag] = []
            self.flags[flag].append(flags[flag])
    
    def setPrimary(self, name):
        """
        Set the primary protocol
        :param name: Name of the primary protocol
        """
        self.primary = name


class tsDatabase:
    """
        A class to represent a time series database. Overall this class will a database style of indexing of files.
        The user will provide a experimental structure, and the class will provide a simple interface to load the data. 
        Each DB entry will be a single cell, it will have metadata describing the cell, and linking the different files to the cell.
        It will also have a table of features, and a table of time series data (hot-loaded from the files).
        
    """
    ## internally this looks something like
    ## Subject | Protocol 1 | Protocol 2 | ... | Protocol N | Metadata | Features | Time Series
    ## 1       | file1      | file2      | ... | fileN      | meta1    | feat1    | ts1
    ## 2       | file1      | file2      | ... | fileN      | meta2    | feat2    | ts2
    ## ...     | ...        | ...        | ... | ...        | ...      | ...      | ...

    def __init__(self):
        """
        Constructor for the tsDatabase class
        :param path: Path to the database
        :param exp: Experimental structure of the database
        """

    def addEntry(self, name, path):
        """
        Add an entry to the database
        :param name: Name of the entry
        :param path: Path to the entry
        """
        pass
            