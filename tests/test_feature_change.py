import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os


def compare_excel_files(file1, file2, sheet_name='full sheet'):
    df1 = pd.read_excel(file1, sheet_name =sheet_name)
    df2 = pd.read_excel(file2, sheet_name =sheet_name)
    #base folder 
    base_folder = os.path.dirname(file1)
    #iter column wise and compare
    unequal_cols = []
    diff = []
    for col in df1.columns:
        if col in df2.columns:
            #if its a numeric dtype
            if np.issubdtype(df1[col].dtype, np.number):
                col1 = df1[col].to_numpy()
                col2 = df2[col].to_numpy()
                #compute the mean perecentage error
                try:
                    mpe = np.nanmean(np.abs(col1 - col2)/col1)
                except:
                    mpe = np.nan
                if mpe < 0.01:
                    print(col, "is equal")
                else:
                    print(f"WARNING: {col} is not equal; PLEASE CHECK, mean percent error is {mpe*100}")
                    unequal_cols.append(col)
                    diff.append(mpe)
            else:
                #check if they are equal using pandas
                if df1[col].equals(df2[col]):
                    print(col, "is equal")
                else:
                    print(f"WARNING: {col} is not equal; PLEASE CHECK")
                    unequal_cols.append(col)
                    diff.append(np.nan)
        else:
            print(f"WARNING: {col} is not in {file2}")

    #output the unequal columns as a csv
    if len(unequal_cols) > 0:
        df_unequal = pd.DataFrame(np.vstack([unequal_cols, diff]).T, columns=['unequal_cols', 'mean percent error'])
        df_unequal.to_csv(base_folder+'//unequal_cols.csv')
        #isolate the unequal columns from df1 and df2
        df_unequal = df_unequal['unequal_cols'].tolist()
        df1_unequal = df1[df_unequal]
        df2_unequal = df2[df_unequal]
        #write the unequal columns to excel
        writer = pd.ExcelWriter(base_folder+'//unequal_cols.xlsx')
        df1_unequal.to_excel(writer, 'df1_unequal')
        df2_unequal.to_excel(writer, 'df2_unequal')
        writer.save()
        print("unequal columns written to excel")


if __name__ == '__main__':
    #prompt the user for excel file 1
    root = tk.Tk()
    root.withdraw()
    file1 = tk.filedialog.askopenfilename()
    #prompt the user for excel file 2
    root = tk.Tk()
    root.withdraw()
    file2 = tk.filedialog.askopenfilename()



    compare_excel_files(file1, file2)
    input('Press ENTER to exit')