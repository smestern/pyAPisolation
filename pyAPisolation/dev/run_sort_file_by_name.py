#%%
import pandas as pd
import numpy as np
import glob
import os
import tkinter as tk
from tkinter import filedialog
import shutil
print("Load finished")
root = tk.Tk()
root.withdraw()

filelist_f = filedialog.askopenfilename(
                                   title='Select File list'
                                   )


search_dir = filedialog.askdirectory(
                                   title='Select search dir'
                                   )
out_dir = filedialog.askdirectory(
                                   title='Select out dir'
                                   )

_sbplot = input("Allow partial file name search?")

try: 
    if _sbplot == 'y' or _sbplot =='Y':
        bplot = True
    else:
        bplot = False
except:
   bplot = False
#%%
filelist = np.genfromtxt(filelist_f, dtype=str, delimiter=',')
if bplot:
    basename_filelist = [os.path.basename(str(x)) for x in filelist]
else:
    basename_filelist = filelist

#%%

print('Searching for Files')
#glob the dir for ABFS
files_in_search = glob.glob(search_dir+'//**/*.abf', recursive=True)
print(f"{len(files_in_search)} abf files found")
#%%
#filtering
for filename in files_in_search:
    any_match = [x in filename for x in basename_filelist]
    
    if np.any(any_match):
        try:
            print(f"Matched to {filename}")
            shutil.copy2(filename, out_dir)
        except:
            print(f"Error copying {filename}")
            pass
# %%

print("==== SUCCESS ====")
input('Press ENTER to exit')
