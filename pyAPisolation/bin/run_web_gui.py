import subprocess
import sys
import os
import voila
import jupyter
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import glob
from ipfx import feature_extractor
import pyabf
file_path = os.path.dirname(os.path.abspath(__file__))

subprocess.run(['voila', file_path+".\\abfanalysis_local.ipynb"])
