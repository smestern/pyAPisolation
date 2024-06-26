import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import glob
import bs4
import json
import sys
from scipy.signal import resample, decimate
from http.server import HTTPServer, CGIHTTPRequestHandler
import matplotlib.pyplot as plt
import anndata as ad
from .webVizConfig import webVizConfig
import shutil
from .flaskApp import tsServer
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class tsDatabase:
    """
    A class to represent a time series database. Overall this class will help you present a tablaur view of metadata and features, whilst also providing a simple interface to load and serve raw time series data.
    Given a dataframe (or equivalent) of time series data, this class will serve the data via a simple web interface, or export the data to a csv, annData object, or static web page.
    takes:
        database_file: a pandas dataframe, anndata obj, or a path to a csv file containing the database
        config: a web_viz_config object, a path to a json file containing the config, or a dictionary containing the config
            config options:
                file_name: the name of the file column in the database
                folder_path: the name of the folder column in the database, the path to the parent directory to find the raw data
                file_path: (optional) the name of the column containing the full path to the raw data
                ext: the file extension of the raw data, e.g. '.abf'
    """
    def __init__(self, database_file, config=None, **kwargs):
        if config is None:
            config = webVizConfig()
        elif isinstance(config, str):
            config = webVizConfig(file=config)
        elif isinstance(config, dict):
            config = webVizConfig(**config)
        config.update(kwargs)
        self.config = config
        #parse the database file
        self.database = None
        self._raw_database = database_file
        self._load_data()
        self._search_for_raw_data()
        
    def export(self):
        #todo
        pass

    def export_to_static_web(self):
        #todo
        pass

    def run(self):
        pass

    #Internal methods
    def _load_data(self):
        """
        Load the database file into a pandas dataframe
        """
        if isinstance(self._raw_database, pd.DataFrame):
            self.database = self._raw_database.copy()
        elif isinstance(self._raw_database, str):
            if self._raw_database.endswith('.csv'): #check if the file is a csv
                self.database = pd.read_csv(self._raw_database).copy()
            elif self._raw_database.endswith('.h5ad'):
                self.database = ad.read(self._raw_database)
        elif isinstance(self._raw_database, ad.AnnData):
            self.database = self._raw_database.obs
        else:
            raise ValueError('Database file must be a pandas dataframe, anndata object, or a path to a csv file')
        self.database = self.database.fillna('')
        #check for file and folder columns, or file path column
        if self.config.file_index not in self.database.columns:
            if self.config.file_path not in self.database.columns:
                raise ValueError(f"At least one of {self.config.file_index} or {self.config.file_path} must be present in the database")
            else:
                self.database[self.config.file_index] = self.database[self.config.file_path].apply(lambda x: os.path.basename(x).split('.')[0])
        elif self.config.file_index in self.database.columns and self.config.file_path not in self.database.columns:
            self.database[self.config.file_path] = self.database[self.config.folder_path].apply(lambda x: os.path.join(x, self.config.file_index)+self.config.ext)
        
        #check that the required columns are present
        for col in self.config.table_vars_rq:
            if col not in self.database.columns:
                logger.warning(f"Column {col} not found in database")

    def _load_data_from_file(self, file):
        #todo
        pass

    def _search_for_raw_data(self):
        #check for raw data. If the raw data is not present, search for it in the folder path
        globbed_files = np.array([glob.glob(os.path.join(x, '*'+self.config.ext)) for x in self.database[self.config.folder_path].unique()])

        logger.info("Searching for raw data")
        for idx, row in self.database.iterrows():
            if not os.path.exists(row[self.config.file_path]):
                raw_data = os.path.join(row[self.config.folder_path], row[self.config.file_index]+self.config.ext)
                if os.path.exists(raw_data):
                    self.database.at[idx, self.config.file_path] = raw_data
                    logger.info(f"Raw data found for {row[self.config.file_index]}")
                else:
                    #if the abs path was not found, search the globbed files
                    found = False
                    for files in globbed_files:
                        if row[self.config.file_index]+self.config.ext in files:
                            self.database.at[idx, self.config.file_path] = files[0]
                            logger.info(f"Raw data found for {row[self.config.file_index]} in the globbed files")
                            found = True
                            break
                    logger.warning(f"Raw data not found for {row[self.config.file_index]}")
        logger.info("Raw data search complete")
        

      