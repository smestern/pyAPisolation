# This Python file uses the following encoding: utf-8
import sys
import os
import glob
import pyabf
import numpy as np
import pandas as pd
import multiprocessing as mp
from sklearn.svm import OneClassSVM
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import copy
from functools import partial
import scipy.signal as signal
from PySide2.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QProgressDialog
from PySide2.QtCore import QFile
from PySide2 import QtGui
import PySide2.QtCore as QtCore

from PySide2.QtUiTools import QUiLoader

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
print("Loaded external libraries")
from pyAPisolation.feature_extractor import folder_feature_extract, save_data_frames, preprocess_abf, analyze_subthres, preprocess_abf_subthreshold
from pyAPisolation.patch_utils import load_protocols
from pyAPisolation.patch_subthres import exp_decay_2p, exp_decay_1p, exp_decay_factor

#import pyqtgraph as pg

import time
from ipfx.feature_extractor import SpikeFeatureExtractor

PLOT_BACKEND = 'matplotlib'
ANALYSIS_TABS = {0:'spike', 1:'subthres'}

class analysis_gui(QWidget):
    def __init__(self):
        super(analysis_gui, self).__init__()
        self.load_ui()
        self.main_widget = self.children()[0]
        self.abf = None
        self.current_filter = 0.
        self.bind_ui()


    def load_ui(self):
        loader = QUiLoader()
        path = os.path.join(os.path.dirname(__file__), "form.ui")
        print(path)
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        loader.load(ui_file, self)
        ui_file.close()

    def bind_ui(self):
        children = self.main_widget.children()
        #assign the children to the main object for easy access
        for child in children:
            print(child.objectName())
            if child.objectName() == "folder_select":
                self.folder_select = child
                #for the file loader make a folder select
                child.clicked.connect(self.file_select)
            elif child.objectName() == "file_list":
                self.file_list = child
                child.itemClicked.connect(self.abf_select)
            elif child.objectName() == "frame":
                self.frame = child
                layout =  QVBoxLayout()
                if PLOT_BACKEND == "pyqtgraph":
                    plot_widget = pg.GraphicsLayoutWidget()
                    layout.addWidget(plot_widget)
                    self.main_view = plot_widget
                elif PLOT_BACKEND == "matplotlib":
                    self.main_view = FigureCanvas(Figure(figsize=(15, 5)))
                    layout.addWidget(self.main_view)
                    self.toolbar = NavigationToolbar(self.main_view, self.frame)
                    layout.addWidget(self.toolbar)
                self.frame.setLayout(layout)
            elif child.objectName() == "sweep_selector":
                self.sweep_selector = child
        #generate the analysis settings listener for spike finder
        self.dvdt_thres = self.main_widget.findChild(QWidget, "dvdt_thres")
        self.dvdt_thres.textChanged.connect(self.analysis_changed)
        self.thres_to_peak_time = self.main_widget.findChild(QWidget, "t_to_p_time")
        self.thres_to_peak_time.textChanged.connect(self.analysis_changed)
        self.thres_to_peak_height = self.main_widget.findChild(QWidget, "t_to_p_height")
        self.thres_to_peak_height.textChanged.connect(self.analysis_changed)
        self.min_peak_height = self.main_widget.findChild(QWidget, "min_peak")
        self.min_peak_height.textChanged.connect(self.analysis_changed)
        self.start_time = self.main_widget.findChild(QWidget, "start")
        self.start_time.textChanged.connect(self.analysis_changed)
        self.end_time = self.main_widget.findChild(QWidget, "end_time")
        self.end_time.textChanged.connect(self.analysis_changed)
        self.protocol_select = self.main_widget.findChild(QWidget, "protocol_selector")
        self.protocol_select.currentIndexChanged.connect(self.change_protocol_select)
        self.bstim = self.main_widget.findChild(QWidget, "bstim")
        self.bessel = self.main_widget.findChild(QWidget, "bessel_filt")
        self.thres_per = self.main_widget.findChild(QWidget, "thres_percent")
        self.thres_per.textChanged.connect(self.analysis_changed)
        #find the output buttons
        self.bspikeFind = self.main_widget.findChild(QWidget, "spikeFinder")
        self.brunningBin = self.main_widget.findChild(QWidget, "runningBin")
        self.brawData = self.main_widget.findChild(QWidget, "rawSpike")

        self.protocol_select.currentIndexChanged.connect(self.analysis_changed)
        self.bessel.textChanged.connect(self.analysis_changed)
        run_analysis = self.main_widget.findChild(QWidget, "run_analysis")
        run_analysis.clicked.connect(self.run_analysis)
        self.refresh = self.main_widget.findChild(QWidget, "refresh_plot")
        self.refresh.clicked.connect(self.analysis_changed_run)
        self.saveCur = self.main_widget.findChild(QWidget, "saveCur")
        self.saveCur.clicked.connect(self._save_csv_for_current_file)
        self.outputTag = self.main_widget.findChild(QWidget, "outputTag")
        #link the settings buttons for the cm calc analysis
        self.tabselect = self.main_widget.findChild(QWidget, "tabWidget")
        self.tabselect.currentChanged.connect(self.analysis_changed_run)
        self.subthresSweeps = self.main_widget.findChild(QWidget, "subthresSweeps")
        self.subthresSweeps.textChanged.connect(self.analysis_changed)
        self.stimPer = self.main_widget.findChild(QWidget, "stimPer")
        self.stimPer.textChanged.connect(self.analysis_changed)
        self.stimfind = self.main_widget.findChild(QWidget, "bstim_2")
        self.stimfind.clicked.connect(self.analysis_changed)
        self.startCM = self.main_widget.findChild(QWidget, "startCM")
        self.startCM.textChanged.connect(self.analysis_changed)
        self.endCM = self.main_widget.findChild(QWidget, "endCM")
        self.endCM.textChanged.connect(self.analysis_changed)
        self.besselFilterCM = self.main_widget.findChild(QWidget, "bessel_filt_cm")
        self.besselFilterCM.textChanged.connect(self.analysis_changed)

        

    
    def file_select(self):
        self.selected_dir = QFileDialog.getExistingDirectory()
        self.abf_list = glob.glob(self.selected_dir + "/**/*.abf", recursive=True)
        self.abf_list_name = [os.path.basename(x) for x in self.abf_list]
        self.pairs = [c for c in zip(self.abf_list_name, self.abf_list)]
        self.abf_file = self.pairs
        self.selected_sweeps = None
        #create a popup about the scanning the files
        self.scan_popup = QProgressDialog("Scanning files", "Cancel", 0, len(self.abf_list))
        self.scan_popup.setWindowModality(QtCore.Qt.WindowModal)
        self.scan_popup.forceShow()
        #Generate the protocol list
        self.protocol_list = []
        self.protocol_file_pair = {}
        for name, abf in self.abf_file:
            try:
                self.scan_popup.setValue(self.scan_popup.value() + 1)
                abf_obj = pyabf.ABF(abf, loadData=False)
                self.protocol_list.append(abf_obj.protocol)
                self.protocol_file_pair[name] = abf_obj.protocol
            except:
                self.protocol_file_pair[name] = 'unknown'
        #we really only care about unique protocols
        #close the popup
        self.scan_popup.close()
        #filter down to unique ones
        self.protocol_list = np.hstack(("[No Filter]", np.unique(self.protocol_list)))
        #clear the file list and protocol select
        self.file_list.clear()
        self.protocol_select.clear()
        self.protocol_select.addItems(self.protocol_list)
        self.file_list.addItems(self.abf_list_name)

    def change_protocol_select(self):
        self.get_selected_protocol()
        #filter the file list by the protocol
        if self.selected_protocol == "[No Filter]"  or self.selected_protocol == "":
            for i in np.arange(self.file_list.count()):
                item = self.file_list.item(i)
                item.setHidden(False)
        else:
            for i in np.arange(self.file_list.count()):
                item = self.file_list.item(i)
                if self.protocol_file_pair[item.text()] != self.selected_protocol:
                    item.setHidden(True)
                else:
                    item.setHidden(False)
        self.analysis_changed()
        

    def abf_select(self, item):
        self.selected_abf = self.abf_file[self.file_list.currentRow()][1]
        self.selected_abf_name = self.abf_file[self.file_list.currentRow()][0]
        #self.selected_protocol = self.protocol_list[self.protocol_select.currentIndex()]
        #for the sweep selector make a checkbox for each sweep
        self.get_selected_abf()

        #delete the children of the old sweep selector
        if self.sweep_selector.layout() is not None:
            self.clear_layout(self.sweep_selector.layout())
            layout = self.sweep_selector.layout()
        else:
            layout = QHBoxLayout() #contains the checkboxes
        
        if (len(self.sweep_selector.children())-1 == self.abf.sweepList) and self.selected_sweeps is not None: #if the sweep list is the same as the sweep list
            bool_toggle = True
        else:
            bool_toggle = False


        self.checkbox_list = []
        for sweep in self.abf.sweepList:
            checkbox = QtWidgets.QCheckBox(str(sweep))
            if bool_toggle:
                if sweep in self.selected_sweeps:
                    checkbox.setChecked(True)
                else:
                    checkbox.setChecked(False)
            else:
                checkbox.setChecked(True)
            checkbox.stateChanged.connect(self.analysis_changed)
            layout.addWidget(checkbox)
            self.checkbox_list.append(checkbox)

        #also add a button to check/uncheck all
        check_all = QtWidgets.QPushButton("Check All")
        check_all.clicked.connect(self.check_all)
        layout.addWidget(check_all)


        if self.sweep_selector.layout() is None:
            self.sweep_selector.setLayout(layout)
        #sometimes the sweep selector is not cleared out
        #idk why 
        self.run_indiv_analysis()
        self.plot_abf()
        
    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget() is not None:
                child.widget().deleteLater()
            elif child.layout() is not None:
                self.clear_layout(child.layout())    

    def plot_abf(self):
        if PLOT_BACKEND == "pyqtgraph":
            self._plot_pyqtgraph()
        elif PLOT_BACKEND == "matplotlib":
            self._plot_matplotlib()
        
    def filter_abf(self, abf):
        #filter the abf with 5 khz lowpass
        if self.bessel.text() == "" or float(self.bessel.text()) < 0:
            return abf
        b, a = signal.bessel(4, float(self.bessel.text()), 'low', norm='phase', fs=abf.dataRate)
        abf.data = signal.filtfilt(b, a, abf.data)
        return abf

    def run_indiv_analysis(self):
        
        #get the current abf
        self.abf = self.get_selected_abf()
        #get the current sweep(s)
        self.get_selected_sweeps()
        self.get_analysis_params()
        if self.get_current_analysis() is 'spike':
            self.subthres_df = None
            if self.selected_sweeps is None:
                self.selected_sweeps = self.abf.sweepList
            #create the spike extractor 
            if self.param_dict['end'] == 0.0 or self.param_dict['end'] > self.abf.sweepX[-1]:
                self.param_dict['end'] = self.abf.sweepX[-1]

            self.spike_extractor = SpikeFeatureExtractor(filter=0,  dv_cutoff=self.param_dict['dv_cutoff'],
                max_interval=self.param_dict['max_interval'], min_height=self.param_dict['min_height'], min_peak=self.param_dict['min_peak'],
                start=self.param_dict['start'], end=self.param_dict['end'], thresh_frac=self.param_dict['thresh_frac'])
            #extract the spikes and make a dataframe for each sweep
            self.spike_df = {}
            self.rejected_spikes = None #{}
            for sweep in self.selected_sweeps:
                self.abf.setSweep(sweep)
                
                self.spike_df[sweep] = self.spike_extractor.process(self.abf.sweepX,self.abf.sweepY, self.abf.sweepC)
                # self.rejected_spikes[sweep] = pd.DataFrame().from_dict(determine_rejected_spikes(self.spike_extractor, self.spike_df[sweep], self.abf.sweepY, self.abf.sweepX, 
                # self.param_dict)).T
                #self.rejected_spikes = None
            #self.spike_df = pd.concat(self.spike_df)
        elif self.get_current_analysis() is 'subthres':
            self.spike_df = None
            self.rejected_spikes = None
            self.subthres_df, _ = analyze_subthres(self.abf, **self.subt_param_dict)
            
    
    def get_analysis_params(self):
        #build the spike param_dict
        dv_cut = float(self.dvdt_thres.text())
        lowerlim = float(self.start_time.text())
        upperlim = float(self.end_time.text())
        tp_cut = float(self.thres_to_peak_time.text())/1000
        min_cut = float(self.thres_to_peak_height.text())
        min_peak = float(self.min_peak_height.text())
        bstim_find = self.bstim.isChecked()
        bessel_filter = float(self.bessel.text())
        thresh_frac = float(self.thres_per.text())

        self.param_dict = {'filter': 0, 'dv_cutoff':dv_cut, 'start': lowerlim, 'end': upperlim, 'max_interval': tp_cut,
         'min_height': min_cut, 'min_peak': min_peak, 'start': lowerlim, 'end': upperlim, 
        'stim_find': bstim_find, 'bessel_filter': bessel_filter, 'thresh_frac': thresh_frac}
        #build the subthres param_dict
        try:
            subt_sweeps = np.fromstring(self.subthresSweeps.text(), dtype=int, sep=',')
            if len(subt_sweeps) == 0:
                subt_sweeps = None
        except:
            subt_sweeps = None
        try:
            start_sear = float(self.startCM.text())
            end_sear = float(self.endCM.text())
        except:
            start_sear = None
            end_sear = None
        time_after = float(self.stimPer.text())
        
        if start_sear == 0:
            start_sear = None
        if end_sear == 0:
            end_sear = None
        self.subt_param_dict ={'subt_sweeps': subt_sweeps, 'time_after': time_after, 'start_sear': start_sear, 'end_sear': end_sear}
        return self.param_dict

    def analysis_changed(self):
        self.get_analysis_params()
        #update the refresh button to reflect that the analysis has changed
        self.refresh.setText("🔄 Refresh plot (Analysis Changed)")
        
    def analysis_changed_run(self):
        if self.abf is not None:
            self.run_indiv_analysis( )
            self.plot_abf()
            self.refresh.setText("🔄 Refresh plot")

    
    def run_analysis(self):
        #check whether we are running spike or subthres
        if self.get_current_analysis() is 'spike':
            #run the folder analysis
            self.get_analysis_params()
            self.get_selected_protocol()
            #df = folder_feature_extract(self.selected_dir, self.param_dict, False, self.selected_protocol)
            df = self._inner_analysis_loop(self.selected_dir, self.param_dict,  self.selected_protocol)     
            save_data_frames(df[0], df[1], df[2], self.selected_dir, str(time.time())+self.outputTag.text(), self.bspikeFind.isChecked()
                             , self.brunningBin.isChecked(), self.brawData.isChecked())
        elif self.get_current_analysis() is 'subthres':
            self.get_analysis_params()
            self.get_selected_protocol()
            df = self._inner_analysis_loop_subthres(self.selected_dir, self.subt_param_dict,  self.selected_protocol)
        
    def get_current_analysis(self):
        index = self.tabselect.currentIndex()
        self.current_analysis = ANALYSIS_TABS[index]
        return self.current_analysis
        

    def get_selected_protocol(self):
        proto = self.protocol_select.currentText()
        if proto == "[No Filter]":
            self.selected_protocol = ''
        else:
            self.selected_protocol = proto
        return self.selected_protocol
        
    def get_selected_sweeps(self):
        self.selected_sweeps = []
        for sweep in self.checkbox_list:
            if isinstance(sweep, QtWidgets.QCheckBox):
                if sweep.isChecked():
                    self.selected_sweeps.append(int(sweep.text()))
        return self.selected_sweeps
    
    def get_selected_abf(self):
        #ensure the selected abf is loaded or reload and filter it
        if self.abf is None:
            self.abf = self.filter_abf(pyabf.ABF(self.selected_abf))
            self.current_filter = self.param_dict['bessel_filter']
        else:
            if (os.path.abspath(self.abf.abfFilePath) == os.path.abspath(self.selected_abf)) and (self.param_dict['bessel_filter'] == self.current_filter):
                pass
            else:
                
                self.abf = self.filter_abf(pyabf.ABF(self.selected_abf))
                self.current_filter = self.param_dict['bessel_filter']
        return self.abf
        
    def check_all(self):
        for sweep in self.checkbox_list:
            if isinstance(sweep, QtWidgets.QCheckBox):
                sweep.blockSignals(True)
                if sweep.isChecked():
                    sweep.setChecked(False)
                else:
                    sweep.setChecked(True)
                sweep.blockSignals(False)
        self.plot_abf()


    def _save_csv_for_current_file(self):
        #for the current abf run the analysis and save the csv
        self.run_indiv_analysis()
        if self.get_current_analysis() is 'spike':
            dfs = preprocess_abf(self.abf.abfFilePath, copy.deepcopy(self.param_dict), False, '')
            save_data_frames(dfs[1], dfs[0], dfs[2], self.selected_dir, str(time.time())+self.outputTag.text())

    def _find_outliers(self, df):
        outlier_dect = OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        temp_df = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(df.select_dtypes(include=['float']))
        outlier_dect.fit(temp_df)
        labels = outlier_dect.predict(temp_df)
        return labels


    def _inner_analysis_loop(self, folder, param_dict, protocol_name):
        debugplot = 0
        running_lab = ['Trough', 'Peak', 'Max Rise (upstroke)', 'Max decline (downstroke)', 'Width']
        dfs = pd.DataFrame()
        df_spike_count = pd.DataFrame()
        df_running_avg_count = pd.DataFrame()
        filelist = glob.glob(folder + "/**/*.abf", recursive=True)
        popup = QProgressDialog("Operation in progress.", "Cancel", 0, len(filelist), self)
        popup.setWindowModality(QtCore.Qt.WindowModal)
        popup.forceShow()
        spike_count = []
        df_full = []
        df_running_avg = []
        parallel_processing = False
        i = 0

        if parallel_processing:
            def update_iteration(_, i):
                
                popup.setValue(i)
                popup.setLabelText("Processing file " + str(i) + " of " + str(len(filelist)))
            pool = mp.Pool(mp.cpu_count())

            results = [pool.apply_async(preprocess_abf, args=(file, copy.deepcopy(param_dict), False, protocol_name), callback=partial(update_iteration, i=i)) for (i, file) in enumerate(filelist)]
            
            ##split out the results
            pool.close()
            #pool.join()
            for result in results:
                popup.setValue(popup.value())
                result.wait()
                temp_res = result.get()
                df_full.append(temp_res[1])
                df_running_avg.append(temp_res[2])
                spike_count.append(temp_res[0])
            pool.join()
        else:
            for i, f in enumerate(filelist):
                popup.setValue(i)
                temp_df_spike_count, temp_full_df, temp_running_bin = preprocess_abf(f, copy.deepcopy(param_dict), False, protocol_name)
                spike_count.append(temp_df_spike_count)
                df_full.append(temp_full_df)
                df_running_avg.append(temp_running_bin)
        df_spike_count = pd.concat(spike_count, sort=True)
        dfs = pd.concat(df_full, sort=True)
        df_running_avg_count = pd.concat(df_running_avg, sort=False)
        popup.hide()
        #detect outliers
        #df_spike_count['outlier'] = self._find_outliers(df_spike_count)
        #Highlight outliers in filelist
        # for i in np.arange(self.file_list.count()):
        #     f = self.file_list.item(i)
        #     if df_spike_count['outlier'].to_numpy()[i] == 1:
        #         f.setBackgroundColor(QtGui.QColor(255, 255, 255))
        #     else:
        #         f.setBackgroundColor(QtGui.QColor(255, 0, 0))

        return dfs, df_spike_count, df_running_avg_count

    def _inner_analysis_loop_subthres(self, folder, param_dict, protocol_name):
        filelist = glob.glob(folder + "/**/*.abf", recursive=True)
        popup = QProgressDialog("Operation in progress.", "Cancel", 0, len(filelist), self)
        popup.setWindowModality(QtCore.Qt.WindowModal)
        popup.forceShow()
        dfs = []
        for i, f in enumerate(filelist):
            popup.setValue(i)
            df = preprocess_abf_subthreshold(f, copy.deepcopy(param_dict), protocol_name)
            dfs.append(df)
        popup.hide()
        return pd.concat(dfs, axis=1)


    def _plot_matplotlib(self):
        self.main_view.figure.clear()
        #self.main_view.figure.canvas.setFixedWidth(900)
        self.axe1 = self.main_view.figure.add_subplot(211)
        self.axe2 = self.main_view.figure.add_subplot(212, sharex=self.axe1)
        #self.main_view.figure.set_facecolor('#F0F0F0')
        #self.main_view.figure.set_edgecolor('#F0F0F0')
       # self.main_view.figure.set_dpi(100)
        self.main_view.figure.set_tight_layout(True)
        #self.main_view.figure.set_facecolor('#F0F0F0')
        self.get_selected_abf()
        self.get_selected_sweeps()
        #for the chosen sweeps
        if self.selected_sweeps == None:
            self.selected_sweeps = self.abf.sweepList

        for sweep in self.selected_sweeps:
            self.abf.setSweep(sweep)
            self.axe1.plot(self.abf.sweepX, self.abf.sweepY, label=str(sweep))
            #plot the dvdt
            self.axe2.plot(self.abf.sweepX[:-1], (np.diff(self.abf.sweepY)/(np.diff(self.abf.sweepX)*1000)))
        self.axe1.set_title(self.selected_abf_name)

        #draw the dvdt threshold
        self.dvdt_thres_value = float(self.dvdt_thres.text())
        self.axe2.axhline(y=self.dvdt_thres_value, color='#FF0000', ls='--')

        #if the analysis has been run, plot the results
        labeled_legend = False
        if self.spike_df is not None:
            for sweep in self.selected_sweeps:
                if self.spike_df[sweep].empty:
                    continue
                
                #plot with labels if its the first sweep
                if labeled_legend == False:
                    self.axe1.scatter(self.spike_df[sweep]['peak_t'], self.spike_df[sweep]['peak_v'], color='#FF0000', s=10, zorder=99, alpha=0.5, label='Spike Peak')
                    self.axe1.scatter(self.spike_df[sweep]['threshold_t'], self.spike_df[sweep]['threshold_v'], color='#00FF00', s=10, zorder=99, alpha=0.5, label='Threshold')
                    #plot the dv/dt threshold
                    try:
                        self.axe2.scatter(self.spike_df[sweep]['downstroke_t'], self.spike_df[sweep]['downstroke'], color='#FF0000', label='Downstroke/Decay')
                        self.axe2.scatter(self.spike_df[sweep]['upstroke_t'], self.spike_df[sweep]['upstroke'], color='#00FF00', label='Upstroke/Rise')
                        labeled_legend = True
                    except:
                        pass
                else:
                    self.axe1.scatter(self.spike_df[sweep]['peak_t'], self.spike_df[sweep]['peak_v'], color='#FF0000', s=10, zorder=99)
                    self.axe1.scatter(self.spike_df[sweep]['threshold_t'], self.spike_df[sweep]['threshold_v'], color='#00FF00', s=10, zorder=99)
                    try:
                        self.axe2.scatter(self.spike_df[sweep]['downstroke_t'], self.spike_df[sweep]['downstroke'], color='#FF0000')
                        self.axe2.scatter(self.spike_df[sweep]['upstroke_t'], self.spike_df[sweep]['upstroke'], color='#00FF00')
                    except:
                        pass
        
        #plot the rejected spikes if they exist
        if self.rejected_spikes is not None:
            #
            
            #create a cmap for the labels
            cmap = plt.cm.get_cmap('viridis')
            #create a list of colors for the labels
            labels = [str(row.to_dict()) for sweep in self.selected_sweeps for index, row in self.rejected_spikes[sweep].iterrows()]
            colors = [cmap(i) for i in np.linspace(0, 1, len(labels))]
            #create a dictionary of colors for the labels
            color_dict = dict(zip(labels, colors))

            for sweep in self.selected_sweeps:
                self.abf.setSweep(sweep)
                if self.rejected_spikes[sweep].empty:
                    continue
                for index, row in self.rejected_spikes[sweep].iterrows():
                    #plot the rejected spikes
                    self.axe1.scatter(self.abf.sweepX[int(index)], self.abf.sweepY[int(index)], color=color_dict[str(row.to_dict())], s=10, zorder=99,)
                    
                    #self.axe1.scatter(self.abf.sweepX[int(index)], self.abf.sweepY[int(index)], c='r', s=10, zorder=99, label=)
           
            #create a legend for the rejected spikes
            #create a list of patches for the legend
            patches = [mpatches.Patch(color=color_dict[label], label=label) for label in color_dict]
            #create the legend
            reject_spikes_legend = self.axe1.legend(handles=patches, loc='upper left', fontsize=8)

        #if the analysis was subthreshold, we need to plot the results
        if self.subthres_df is not None:
            cols = self.subthres_df.columns
            for sweep in self.selected_sweeps:
                self.abf.setSweep(sweep)
                if sweep < 9:
                    real_sweep_number = '00' + str(sweep + 1)
                elif sweep > 8 and sweep < 99:
                    real_sweep_number = '0' + str(sweep + 1)
                cols_for_sweep = [c for c in cols if real_sweep_number in c]
                if len(cols_for_sweep) == 0:
                    continue
                temp_df = self.subthres_df[cols_for_sweep]
                #decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, p_decay = exp_decay_factor(dataT, dataV, dataI, time_after, abf_id=abf.abfID)
                #pull out the params, we want the decay, A1, b1, b2
                decay_fast = 1/temp_df[f"fast 2 phase decay {real_sweep_number}"].to_numpy()[0]
                decay_slow = 1/temp_df[f"slow 2 phase decay {real_sweep_number}"].to_numpy()[0]
                a = temp_df[f"Curve fit A {real_sweep_number}"].to_numpy()[0]
                b1 = temp_df[f"Curve fit b1 {real_sweep_number}"].to_numpy()[0]
                b2 = temp_df[f"Curve fit b2 {real_sweep_number}"].to_numpy()[0]
                #compute the curve fit
                dataT, dataV, dataI = self.abf.sweepX, self.abf.sweepY, self.abf.sweepC
                time_aft = 0.5
                diff_I = np.diff(dataI)
                downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
                end_index = downwardinfl + int((np.argmax(diff_I)- downwardinfl) * time_aft)

                upperC = np.amax(dataV[downwardinfl:end_index])
                lowerC = np.amin(dataV[downwardinfl:end_index])
                diff = np.abs(upperC - lowerC)
                t1 = dataT[downwardinfl:end_index] - dataT[downwardinfl]
                y = exp_decay_2p(t1, a,b1, decay_fast, b2, decay_slow)
                self.axe1.plot(dataT[downwardinfl:end_index], y, color='#00FF00', zorder=99)
                #also plot the sag
                upwardinfl = np.nonzero(np.where(diff_I>0, diff_I, 0))[0][0]
                
                diff_I = np.diff(dataI)
                downwardinfl = np.nonzero(np.where(diff_I<0, diff_I, 0))[0][0]
                end_index2 = upwardinfl - int((upwardinfl - downwardinfl) * time_aft)
                dt = dataT[1] - dataT[0] #in s
                vm = np.nanmean(dataV[end_index:upwardinfl])
         
                min_point = downwardinfl + np.argmin(dataV[downwardinfl:end_index2])
                
                avg_min = np.nanmean(dataV[min_point])
                sag_diff = avg_min - vm
                sag_diff_plot = np.arange(avg_min, vm, 1)
                self.axe1.scatter(dataT[min_point], dataV[min_point], c='r', marker='x', zorder=99, label="Min Point")
                self.axe1.scatter(dataT[end_index:upwardinfl], dataV[end_index:upwardinfl], c='g', zorder=99, label="Mean Vm Measured")
                self.axe1.plot(dataT[np.full(sag_diff_plot.shape[0], min_point, dtype=np.int64)], sag_diff_plot, label=f"Sag of {sag_diff}")


                
        self.axe1.legend( bbox_to_anchor=(1.05, 1),
                         loc='upper left', borderaxespad=0.)
        self.axe2.legend(loc='upper right')
        #self.axe1.add_artist(reject_spikes_legend)
        self.main_view.draw()

    def _plot_pyqtgraph(self):
        '''TODO'''
        self.main_view.clear()
        #self.main_view.figure.canvas.setFixedWidth(900)
        self.axe1 = self.main_view.addPlot(1,1)
        self.axe2 = self.main_view.addPlot(2,1)
        #self.main_view.figure.set_facecolor('#F0F0F0')
        #self.main_view.figure.set_edgecolor('#F0F0F0')
       # self.main_view.figure.set_dpi(100)
        #s#elf.main_view.figure.set_tight_layout(True)
        #self.main_view.figure.set_facecolor('#F0F0F0')
        self.get_selected_abf()
        self.get_selected_sweeps()
        #for the chosen sweeps
        if self.selected_sweeps == None:
            self.selected_sweeps = self.abf.sweepList

        for sweep in self.selected_sweeps:
            self.abf.setSweep(sweep)
            self.axe1.plot(self.abf.sweepX, self.abf.sweepY, color='#000000')
            #plot the dvdt
            self.axe2.plot(self.abf.sweepX[:-1], (np.diff(self.abf.sweepY)/np.diff(self.abf.sweepX))/1000)
        #self.axe1.set_title(self.selected_abf_name)

        #draw the dvdt threshold
        self.dvdt_thres_value = float(self.dvdt_thres.text())
        #self.axe2.axhline(y=self.dvdt_thres_value, color='#FF0000', ls='--')

        #if the analysis has been run, plot the results
        if self.spike_df is not None:
            for sweep in self.selected_sweeps:
                if self.spike_df[sweep].empty:
                    continue
                #self.axe1.scatter(self.spike_df[sweep].loc[:, 'peak_t'], self.spike_df[sweep].loc[:,'peak_v'], color='#FF0000', s=10, zorder=99)

from ipfx import spike_detector,time_series_utils
def determine_rejected_spikes(spfx, spike_df, v, t, param_dict):
    """Determine which spikes were rejected by the spike detection algorithm.
    Parameters
    ----------
    spfx : SweepFeatures object
    spike_df : pandas.DataFrame
        DataFrame containing spike features
    Returns
    -------
    rejected_spikes : list of bool
        True if spike was rejected, False if spike was accepted
    """
    dvdt = time_series_utils.calculate_dvdt(v, t, 0)

    rejected_spikes = {}
    intial_spikes = spike_detector.detect_putative_spikes(v, t, param_dict['start'], param_dict['end'],
                                                    dv_cutoff=param_dict['dv_cutoff'],
                                                    dvdt=dvdt)
    peaks = spike_detector.find_peak_indexes(v, t, intial_spikes, param_dict['end'])
    if len(peaks) == 0:
        return rejected_spikes
    diff_mask = [np.any(dvdt[peak_ind:spike_ind] < 0)
                 for peak_ind, spike_ind
                 in zip(peaks[:-1], intial_spikes[1:])]
    peak_indexes = peaks[np.array(diff_mask + [True])]
    spike_indexes = intial_spikes[np.array([True] + diff_mask)]

    peak_level_mask = v[peak_indexes] >= param_dict['min_peak']
        

    height_mask = (v[peak_indexes] - v[spike_indexes]) >= param_dict['min_height']
    for i, spike in enumerate(peak_indexes):
        if np.any([~peak_level_mask[i], ~height_mask[i]]):
            rejected_spikes[spike] = {'peak_level': ~peak_level_mask[i], 'height': height_mask[i]}
    
    peak_level_mask = v[peak_indexes] >= param_dict['min_peak']
    spike_indexes = spike_indexes[peak_level_mask]
    peak_indexes = peak_indexes[peak_level_mask]

    height_mask = (v[peak_indexes] - v[spike_indexes]) >= param_dict['min_height']
    spike_indexes = spike_indexes[height_mask]
    peak_indexes = peak_indexes[height_mask]
    
    if len(spike_indexes) == 0:
        return rejected_spikes
    upstroke_indexes = spike_detector.find_upstroke_indexes(v, t, spike_indexes, peak_indexes, filter=0, dvdt=dvdt)
    thresholds = spike_detector.refine_threshold_indexes(v, t, upstroke_indexes, param_dict['thresh_frac'],
                                                dvdt=dvdt)


    # overlaps = np.flatnonzero(spike_indexes[1:] <= peak_indexes[:-1] + 1)
    # if overlaps.size:
    #     spike_mask = np.ones_like(spike_indexes, dtype=bool)
    #     spike_mask[overlaps + 1] = False
    #     spike_indexes = spike_indexes[spike_mask]

    #     peak_mask = np.ones_like(peak_indexes, dtype=bool)
    #     peak_mask[overlaps] = False
    #     peak_indexes = peak_indexes[peak_mask]

    #     upstroke_mask = np.ones_like(upstroke_indexes, dtype=bool)
    #     upstroke_mask[overlaps] = False
    #     upstroke_indexes = upstroke_indexes[upstroke_mask]

    # Validate that peaks don't occur too long after the threshold
    # If they do, try to re-find threshold from the peak
    too_long_spikes = []
    for i, (spk, peak) in enumerate(zip(spike_indexes, peak_indexes)):
        if t[peak] - t[spk] >= param_dict['max_interval']:
            too_long_spikes.append(i)
    if too_long_spikes:
        i
        avg_upstroke = dvdt[upstroke_indexes].mean()
        target = avg_upstroke * param_dict['thresh_frac']
        drop_spikes = []
        for i in too_long_spikes:
            # First guessing that threshold is wrong and peak is right
            peak = peak_indexes[i]
            t_0 = time_series_utils.find_time_index(t, t[peak] - param_dict['max_interval'])
            below_target = np.flatnonzero(dvdt[upstroke_indexes[i]:t_0:-1] <= target)
            if not below_target.size:
                # Now try to see if threshold was right but peak was wrong

                # Find the peak in a window twice the size of our allowed window
                spike = spike_indexes[i]
                t_0 = time_series_utils.find_time_index(t, t[spike] + 2 * param_dict['max_interval'])
                new_peak = np.argmax(v[spike:t_0]) + spike

                # If that peak is okay (not outside the allowed window, not past the next spike)
                # then keep it
                if t[new_peak] - t[spike] < param_dict['max_interval'] and \
                   (i == len(spike_indexes) - 1 or t[new_peak] < t[spike_indexes[i + 1]]):
                    peak_indexes[i] = new_peak
                else:
                    # Otherwise, log and get rid of the spike
                    drop_spikes.append(i)
#                     raise FeatureError("Could not redetermine threshold")
            else:
                spike_indexes[i] = upstroke_indexes[i] - below_target[0]
        for i in drop_spikes:
            rejected_spikes[spike_indexes[i]] = {'peak_level': False, 'height': False, 'threshold to peak': True, }
    else:
        return rejected_spikes
    return rejected_spikes



if __name__ == "__main__":
    mp.freeze_support()
    app = QApplication([])
    widget = analysis_gui()
    widget.show()
    sys.exit(app.exec_())
