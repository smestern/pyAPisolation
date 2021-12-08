import numpy as np
import matplotlib.pyplot as plt
import pyabf
import tkinter as tk
from tkinter import filedialog
from pyabf import filter
import os
import zipfile
import shutil
from scipy.stats import mode
from scipy import signal
import pandas as pd

root = tk.Tk()
root.withdraw()


dir_path = filedialog.askdirectory(title="Choose Dir to sort")
out_path = filedialog.askdirectory(title="Choose output dir")

def find_or_create_folder(out_path, label):

    full_path = os.path.join(out_path, label)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path

for root,dirs,files in os.walk(dir_path): 
    for x in files:
        fp = os.path.join(root, x)

        if '.abf' in x:
            print(f"opening {fp}")
            try:
                abf = pyabf.ABF(fp, loadData=False)
                proto = abf.protocol
                new_path = find_or_create_folder(out_path, proto)
                print(f"copying {fp}")
                shutil.copy2(fp, new_path)
            except:
                pass
            


