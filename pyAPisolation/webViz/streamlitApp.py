import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import bs4
import json
import sys
import logging
from scipy.signal import resample, decimate
from pyAPisolation.patch_ml import *
from pyAPisolation.patch_utils import *
from pyAPisolation.featureExtractor import *
from pyAPisolation.loadFile import loadABF
import pyabf
from http.server import HTTPServer, CGIHTTPRequestHandler
import matplotlib.pyplot as plt
import anndata as ad
import streamlit as st
import argparse

import shutil
from .flaskApp import tsServer
from .tsDatabaseViewer import tsDatabaseViewer
from .webVizConfig import webVizConfig

logger = logging.getLogger(__name__)

#arg parse for command line use
def parse_args():
    parser = argparse.ArgumentParser(description='A simple tool to view and analyze time series data')
    parser.add_argument('database', type=str, help='path to the database file')
    parser.add_argument('--config', type=str, help='path to the config file')
    parser.add_argument('--port', type=int, help='port to serve the data on', default=8000)
    args = parser.parse_args()
    database = pd.read_csv(args.database)
    if args.config:
        config = webVizConfig(file=args.config)
    else:
        config = webVizConfig()



