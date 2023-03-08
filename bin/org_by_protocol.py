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
from tkinter import *
root = Tk()
#root.withdraw()
dir_path = filedialog.askdirectory(title="Choose Dir to sort")
out_path = filedialog.askdirectory(title="Choose output dir")

##list subfolders
subfold = os.listdir(dir_path)
l = tk.Listbox(root, width = 400,
    selectmode='extended')
l.pack()
[l.insert(END, x) for x in subfold]

subfolders_to_use = []
def close():
    global l, root, subfolders_to_use
    subfolders_to_use = l.curselection()
    
    root.destroy()

b = tk.Button(root, text = "OK", command = close).pack()
root.mainloop()

subfolders_to_use = np.array(subfold)[np.array(subfolders_to_use)]
subfolders_to_use = [os.path.join(dir_path, x) for x in subfolders_to_use]


def find_or_create_folder(out_path, label):

    full_path = os.path.join(out_path, label)
    if not os.path.exists(full_path):
        os.makedirs(full_path)

    return full_path

for dir_paths in subfolders_to_use:
    for root,dirs,files in os.walk(dir_paths): 
        for x in files:
            fp = os.path.join(root, x)

            if '.abf' in x:
                print(f"opening {fp}")
                try:
                    abf = pyabf.ABF(fp, loadData=False)
                    proto = abf.protocol
                    if '\\' in proto:
                        proto = proto.split('\\')[-1]
                    new_path = find_or_create_folder(out_path, proto)
                    print(f"copying {fp}")
                    shutil.copy2(fp, new_path)
                except:
                    pass
            


