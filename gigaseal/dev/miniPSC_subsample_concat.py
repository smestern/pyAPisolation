import os
import PySide2.QtWidgets as QtWidgets
from PySide2.QtWidgets import QFileDialog, QListView, QTreeView, QAbstractItemView
import numpy as np
import pandas as pd
from scipy import stats
import pyabf
import datetime

MINIANALYSIS_HEADERS = ["event_number", "time", 'amplitude', 'rise', 'decay', 'area', 'baseline', 'noise',
                        'group', 'channel', '10-90rise', 'halfwidth', 'rise50', 'peak_dir', 'burst', 'burst_sub', '10-90slope', 'rel_time']


def load_dirs():
    # ask the user to select the directory with pyside2
    app = QtWidgets.QApplication([])
    file_dialog = QFileDialog()
    file_dialog.setFileMode(QFileDialog.DirectoryOnly)
    file_dialog.setOption(QFileDialog.DontUseNativeDialog, True)
    file_view = file_dialog.findChild(QListView, 'listView')

    # to make it possible to select multiple directories: # https://stackoverflow.com/questions/38252419/how-to-get-qfiledialog-to-select-and-return-multiple-folders
    if file_view:
        file_view.setSelectionMode(QAbstractItemView.MultiSelection)
    f_tree_view = file_dialog.findChild(QTreeView)
    if f_tree_view:
        f_tree_view.setSelectionMode(QAbstractItemView.MultiSelection)

    if file_dialog.exec():
        paths = file_dialog.selectedFiles()
    app.quit()

    #okay now we have the paths
    #we need to get the files in each path, also we need to check if the files are in the right format
    #in each path there should be a .csv file titled "{folder_name}_data.csv"
    valid_paths = {}
    for path in paths:
        df_out = verify_files(path)
        if not df_out.empty:
            print(f"Files in {path} are in the right format")
            valid_paths[path] = df_out
        else:
            print(f"Files in {path} are not in the right format")

    return valid_paths

def verify_files(path):
    #check if the files are in the right format
    #in each path there should be a .csv file titled "{folder_name}_data.csv"
   
    files = os.listdir(path)
    dir_name = os.path.basename(path).split("_subsampled")[0]
    if f"{dir_name}_data.csv" in files:
        pass
    else:
        print(f"Error: {dir_name}_data.csv not found in {path}")
        return pd.DataFrame()
    
    sub_dir_df = pd.read_csv(os.path.abspath(f"{path}/{dir_name}_data.csv"), index_col=0)

    #load the primary abf file
    if not f"{dir_name}.abf" in files:
        print(f"Error: {dir_name}.abf not found in {path}")
        return pd.DataFrame()
    else:
        prime_abf = pyabf.ABF(f"{path}/{dir_name}.abf", loadData=False)
        prime_abf_sweeps = prime_abf.sweepCount
        prime_abf_wallclock = prime_abf.abfDateTime

    #delete the primary abf from the list of files
    files.remove(f"{dir_name}.abf")

    #now each subfile (abf) should have a corresponding .asc file

    #add the wallclock times to the sub_dir_df as blank columns
    sub_dir_df["wallclock_start"] = np.nan
    sub_dir_df["wallclock_end"] = np.nan

    for file in files:
        if file.endswith(".abf"):
            #load the abf and get the times
            abf = pyabf.ABF(f"{path}/{file}", loadData=False)

            if not f"{file[:-4]}.ASC" in files:
                print(f"Error: {file[:-4]}.ASC not found in {path}")
            else:
                pass
            new_df = {}
            if file in sub_dir_df.index.values:
                row = sub_dir_df.loc[file]
                
                #we need to add the wallclock to the time_start (in seconds)
                start_times = row["time_start"]
                wallclock_start = prime_abf_wallclock + datetime.timedelta(seconds=start_times)
                
                #repeat for time_end
                end_times = row["time_end"]
                wallclock_end = prime_abf_wallclock + datetime.timedelta(seconds=end_times)

                sub_dir_df.loc[file, "wallclock_start"] = wallclock_start
                sub_dir_df.loc[file, "wallclock_end"] = wallclock_end
                
    
    return sub_dir_df


def process_folder(path, sub_dir_df):
    #load the primary abf file
    files = os.listdir(path)
    dir_name = os.path.basename(path).split("_subsampled")[0]

    #iter through the sub_dir_df, which are the subfiles
    event_frames = []
    for index, row in sub_dir_df.iterrows():
        #strip the .abf from the index
        abf = pyabf.ABF(f"{path}/{index}", loadData=False)
        if not os.path.exists(f"{path}/{index[:-4]}.ASC"):
            print(f"Error: {index[:-4]}.ASC not found in {path}")
            continue
        asc = pd.read_csv(f"{path}/{index[:-4]}.ASC", sep="\t", header=None)
        asc.columns = MINIANALYSIS_HEADERS
        #time is in ms, we need to convert to seconds
        asc['time_s'] = asc['time'].map(lambda x: datetime.timedelta(seconds=float(x.replace(',',''))/1000))
        #SORT BY TIME
        asc = asc.sort_values(by='time_s')

        #now add the wallclock to the time
        asc['time_wallclock'] = row["wallclock_start"] + asc['time_s']
        #add the overall event freq as a column
        asc['inst_event_freq'] = 1/(asc['time_wallclock'] - asc['time_wallclock'].shift(1)).dt.total_seconds()

        event_count_total = len(asc)
        #add the event count as a column
        asc['event_count'] = np.full(event_count_total, event_count_total)
        asc['overall_event_freq'] = np.full(event_count_total, event_count_total/(asc['time_wallclock'].max() - asc['time_wallclock'].min()).total_seconds())

        #add the filename as a column
        asc['filename'] = index

        event_frames.append(asc)
    #concatenate the event frames
    event_frame = pd.concat(event_frames)

    

    return event_frame


if __name__=="__main__":

    BIN_SIZE = 240

    dirs = load_dirs()
    event_frames = {}
    for path, sub_dir_df in dirs.items():
        #save the sub_dir_df to the path
        sub_dir_df.to_csv(os.path.join(path, f"{os.path.basename(path).split('_subsampled')[0]}_data_process.csv"))
        frame = process_folder(path, sub_dir_df)
        event_frames[path] = frame


    #concat all event frames assuming theya are all from one exp
    all_frames = pd.concat(event_frames)
    #make a new column of experiment time that baselines wallclock time to the first event
    all_frames['exp_time'] = (all_frames['time_wallclock'] - all_frames['time_wallclock'].min()).dt.total_seconds()
    out_path = os.path.join(os.path.dirname(list(dirs.keys())[0]), "all_events.csv")
    all_frames.to_csv(out_path)
    #now we can do some analysis on all_frames

    #we want to make a running average of the events with 1 minute bins
    #we can do this by using the pd.cut function
    #we will also make a column of the number of events in each bin
    all_frames['time_bin'] = pd.cut(all_frames['exp_time'], np.arange(0, all_frames['exp_time'].max(), BIN_SIZE))
    
    analysis_frames = all_frames.groupby('time_bin').mean()
    analysis_frames['num_events'] = all_frames.groupby('time_bin').size()
    #add in a column of the time in minutes
    analysis_frames['time_min'] = analysis_frames.index.map(lambda x: x.left/60)
    #add in the std for each column
    for col in analysis_frames.columns:
        if col in ['time_min', 'num_events']:
            continue
        analysis_frames[f"{col}_std"] = all_frames.groupby('time_bin').std()[col]
    

    analysis_frames.to_csv(os.path.join(os.path.dirname(list(dirs.keys())[0]), "analysis.csv"))
