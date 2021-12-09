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
import chart_studio.plotly

dir_name = os.path.dirname(os.path.abspath(__file__))
path_nb = dir_name +"\\abfanalysis_local.ipynb"
print(f"running Server")
result = subprocess.call(
    [sys.executable, '-m', 'voila', path_nb, '--debug']
)
print(f"server Started")
print("stdout:", result.stdout)
