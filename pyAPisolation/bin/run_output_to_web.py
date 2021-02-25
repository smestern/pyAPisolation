import numpy as np
import pandas as pd
import os
import tkinter as tk
from tkinter import filedialog
import bs4
import json
index_col = "__a_filename"

files = filedialog.askopenfilenames(filetypes=(('ABF Files', '*.csv'),
                                   ('All files', '*.*')),
                                   title='Select Input File'
                                   )
fileList = files
full_dataframe = pd.DataFrame()
for x in fileList:
    temp = pd.read_csv(x, )
    full_dataframe = full_dataframe.append(temp)
#full_dataframe = full_dataframe.set_index(index_col)
#full_dataframe = full_dataframe.select_dtypes(["float32", "float64", "int32", "int64"])
full_dataframe = full_dataframe.drop(labels=['Unnamed: 0'], axis=1)
print(full_dataframe)
json_df = full_dataframe.to_json(orient='records')
parsed = json.loads(json_df)
json_str = json.dumps(parsed, indent=4)  
with open("template.html") as inf:
    txt = inf.read()
    soup = bs4.BeautifulSoup(txt, 'html.parser')

json_var = '  var data_tb = ' + json_str + ' '

tag = soup.new_tag("script")
tag.append(json_var)

head = soup.find('body')


head.insert_before(tag)

with open("output.html", "w") as outf:
    outf.write(str(soup))