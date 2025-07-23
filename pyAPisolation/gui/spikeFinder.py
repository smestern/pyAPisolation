# This Python file uses the following encoding: utf-8
import sys
import os
import glob
import pyabf
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib
from  .mainwindow import Ui_MainWindow
matplotlib.use('QtAgg')

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
import copy
from functools import partial
import scipy.signal as signal
print("Loaded basic libraries; importing QT")
from PySide2.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QProgressDialog, QAction
from PySide2.QtCore import QFile, QAbstractTableModel, Qt, QModelIndex
from PySide2 import QtGui
import PySide2.QtCore as QtCore
from PySide2.QtUiTools import QUiLoader
print("Loaded QT libraries")
from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
from matplotlib.widgets import SpanSelector
print("Loaded external libraries")
#import pyAPisolation
from pyAPisolation.featureExtractor import save_data_frames, save_subthres_data, \
process_file, analyze_subthres, preprocess_abf_subthreshold, determine_rejected_spikes
from pyAPisolation.patch_subthres import exp_decay_2p
from pyAPisolation.patch_utils import sweepNumber_to_real_sweep_number
from pyAPisolation.dev.prism_writer_gui import PrismWriterGUI
import time
from ipfx.feature_extractor import SpikeFeatureExtractor

PLOT_BACKEND = 'matplotlib'
if PLOT_BACKEND == "pyqtgraph":
    import pyqtgraph as pg
    pg.setConfigOptions(imageAxisOrder='row-major', background='w', useNumba=True, useOpenGL=True)
ANALYSIS_TABS = {0:'spike', 1:'subthres'}

class analysis_gui(object):
    def __init__(self, app: QApplication):
        #super(analysis_gui, self).__init__()
        self.app = app
        self.load_ui()
        #self.main_widget = self.children()[-1]
        self.abf = None
        self.current_filter = 0.
        self.bind_ui()

    def load_ui(self):
        # loader = QUiLoader()
        # path = os.path.join(os.path.dirname(__file__), "mainwindowMDI.ui")
        # ui_file = QFile(path)
        # ui_file.open(QFile.ReadOnly)
        self.main_window = QtWidgets.QMainWindow()
        self.main_widget = Ui_MainWindow().setupUi(self.main_window) #loader.load(ui_file)
        #ui_file.close()

    def bind_ui(self):
        #assign the children to the main object for easy access
        self.folder_select = self.main_widget.findChild(QWidget, "folder_select")
        self.folder_select.clicked.connect(self.file_select)
        self.mdi_area = self.main_widget.findChild(QWidget, "mdiArea")
        
        self.file_list = self.main_widget.findChild(QWidget, "file_list")
        self.file_list.itemClicked.connect(self.abf_select)
        
        self.frame = self.main_widget.findChild(QWidget, "mainplot")
        layout = QVBoxLayout()
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
        
        self.sweep_selector = self.main_widget.findChild(QWidget, "sweep_selector")
        #generate the analysis settings listener for spike finder
        self.dvdt_thres = self.main_widget.findChild(QWidget, "dvdt_thres")
        self.dvdt_thres.editingFinished.connect(self.analysis_changed)
        self.thres_to_peak_time = self.main_widget.findChild(QWidget, "t_to_p_time")
        self.thres_to_peak_time.editingFinished.connect(self.analysis_changed)
        self.thres_to_peak_height = self.main_widget.findChild(QWidget, "t_to_p_height")
        self.thres_to_peak_height.editingFinished.connect(self.analysis_changed)
        self.min_peak_height = self.main_widget.findChild(QWidget, "min_peak")
        self.min_peak_height.editingFinished.connect(self.analysis_changed)
        self.start_time = self.main_widget.findChild(QWidget, "start")
        self.start_time.editingFinished.connect(self.analysis_changed)
        self.end_time = self.main_widget.findChild(QWidget, "end_time")
        self.end_time.editingFinished.connect(self.analysis_changed)
        self.protocol_select = self.main_widget.findChild(QWidget, "protocol_selector")
        self.protocol_select.currentIndexChanged.connect(self.change_protocol_select)
        self.bstim = self.main_widget.findChild(QWidget, "bstim")
        self.bessel = self.main_widget.findChild(QWidget, "bessel_filt")
        self.thres_per = self.main_widget.findChild(QWidget, "thres_percent")
        self.thres_per.editingFinished.connect(self.analysis_changed)
        #find the output buttons
        self.bspikeFind = self.main_widget.findChild(QWidget, "spikeFinder")
        self.brunningBin = self.main_widget.findChild(QWidget, "runningBin")
        self.brawData = self.main_widget.findChild(QWidget, "rawSpike")

        self.protocol_select.currentIndexChanged.connect(self.analysis_changed)
        self.bessel.editingFinished.connect(self.analysis_changed)
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
        self.subthresSweeps.editingFinished.connect(self.analysis_changed)
        self.stimPer = self.main_widget.findChild(QWidget, "stimPer")
        self.stimPer.editingFinished.connect(self.analysis_changed)
        self.stimfind = self.main_widget.findChild(QWidget, "bstim_2")
        self.stimfind.clicked.connect(self.analysis_changed)
        self.startCM = self.main_widget.findChild(QWidget, "startCM")
        self.startCM.editingFinished.connect(self.analysis_changed)
        self.endCM = self.main_widget.findChild(QWidget, "endCM")
        self.endCM.editingFinished.connect(self.analysis_changed)
        self.besselFilterCM = self.main_widget.findChild(QWidget, "bessel_filt_cm")
        self.besselFilterCM.editingFinished.connect(self.analysis_changed)

        #Find the table view
        self.tableView = self.main_widget.findChild(QWidget, "resultsTable")
        #make sortable
        self.tableView.setSortingEnabled(True)
        #bind click events to the table view
        self.tableView.clicked.connect(self._results_table_select)

        #the top dropdowns
        self.actionEnable_Parallel = self.main_widget.findChild(QAction, "actionEnable_Parallel")
        self.actionEnable_Parallel.triggered.connect(self.analysis_changed)
        #self.actionEnable_Parallel.triggered.connect(self.analysis_changed)
        self.actionShow_Rejected_Spikes = self.main_widget.findChild(QAction, "actionShow_Rejected_Spikes")
        self.actionShow_Rejected_Spikes.triggered.connect(self.analysis_changed)

        self.actionOpen_Folder = self.main_widget.findChild(QAction, "actionOpen_Folder")
        self.actionOpen_Folder.triggered.connect(self.file_select)

        self.actionOrganize_Subthres = self.main_widget.findChild(QAction, "actionOpen_Results")
        self.actionOrganize_Subthres.triggered.connect(self.results_select)

        self.actionOpen_Results = self.main_widget.findChild(QAction, "actionExit")
        self.actionOpen_Results.triggered.connect(self.exit)

        self.actionOrganize_Abf = self.main_widget.findChild(QAction, "actionOrganize_Abf")
        self.actionOrganize_Abf.triggered.connect(lambda x: self._run_script(False, name='actionOrganize_Abf'))

        

        #for all the windows in the mdi, we want to add a listener for the close event
        self.mdi = self.main_widget.findChild(QWidget, "mdiArea")
        #we also want to add a view button to the dropdown for each window in the mdi
        #eg. view -> window 1, view -> window 2
        #add it programatically
        self.topBar = self.main_widget.findChild(QWidget, "menubar")
        self.viewBar = self.topBar.addMenu("View")
        #tools menu
        self.tools_menu = self.topBar.findChild(QWidget, "menuTools")
        #add a seperator
        self.viewBar.addSeparator()
        for sub in self.mdi.subWindowList():
            self.viewBar.addAction(sub.windowTitle())
            self.viewBar.triggered.connect(self._view_window)
            #set hide on close
            sub.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
            #delete the close button
            sub.setWindowFlags(QtCore.Qt.WindowMinMaxButtonsHint)
        self.viewBar.addSeparator()
        #add a action here to spawn the prism writer
        self.actionPrism_Writer = self.viewBar.addAction("Prism Writer")
        self.actionPrism_Writer2 = self.tools_menu.addAction("Prism Writer")
        self.actionPrism_Writer.triggered.connect(self._prism_writer)
        self.actionPrism_Writer2.triggered.connect(self._prism_writer)


        #plotting windows => 
        self.plot_windows = {}


    def file_select(self):
        """Opens a file dialog to select a folder of abf files
        Takes:
            files: list of abf files
        Returns:
            None
        """
        self.selected_dir = QFileDialog.getExistingDirectory()
        self.abf_list = glob.glob(self.selected_dir + "/**/*.abf", recursive=True)
        self.abf_list_name = [os.path.basename(x) for x in self.abf_list]
        self.pairs = [c for c in zip(self.abf_list_name, self.abf_list)]
        self.abf_file = self.pairs
        self.selected_sweeps = None
        #create a popup about the scanning the files
        self.scan_popup = QProgressDialog("Scanning files", "Cancel", 0, len(self.abf_list), parent=None)
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
        self.scan_popup.deleteLater()
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
        
    def results_select(self):
        self.selected_file = QFileDialog.getOpenFileName(self.main_widget, "Open Excel", filter="Excel Files (*.csv, *.xlsx)")
        self.selected_file = self.selected_file[0]
        self.df = pd.read_csv(self.selected_file) if self.selected_file.endswith('.csv') else pd.read_excel(self.selected_file)
        self.tableView.setModel(PandasModel(self.df, index='filename', parent=self.tableView))

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
            
        #finally if the time constraints are outside the bounds of the abf, reset them
        if float(self.start_time.text()) > self.abf.sweepX[-1]:
            self.start_time.setValue(0)
        if float(self.end_time.text()) > self.abf.sweepX[-1]:
            self.end_time.setValue(self.abf.sweepX[-1])

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
        #check if the filter is below the nyquist frequency
        if float(self.bessel.text()) > abf.dataRate/2:
            #create a popup ask the user if they want to adjust the filter or continue
            pop = QtWidgets.QMessageBox()
            pop.setText("The filter is above the nyquist frequency, do you want to adjust the filter? Otherwise no filter will be applied.")
            pop.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            pop.setDefaultButton(QtWidgets.QMessageBox.No)
            ret = pop.exec_()
            if ret == QtWidgets.QMessageBox.No:
                return abf
            else:
                self.bessel.setValue(float(abf.dataRate/2)-1)

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
        #create a popup for the analysis to warn the user that it may take a while
        self.indiv_popup = QProgressDialog("Operation in progress.", "Cancel", 0, len(self.selected_sweeps), parent=None)
        self.indiv_popup.setWindowModality(QtCore.Qt.WindowModal)
        self.indiv_popup.forceShow()
        show_rejected = self.actionShow_Rejected_Spikes.isChecked()
        if self.get_current_analysis() is 'spike':
            self.subthres_df = None
            if self.selected_sweeps is None:
                self.selected_sweeps = self.abf.sweepList 
            temp_param_dict = copy.deepcopy(self.param_dict) #copy and avoid changing the original, we need to override some settings
            #create the spike extractor 
            if temp_param_dict['end'] == 0.0 or temp_param_dict['end'] > self.abf.sweepX[-1]:
                temp_param_dict['end'] = self.abf.sweepX[-1]

            self.spike_extractor = SpikeFeatureExtractor(filter=0,  dv_cutoff=temp_param_dict['dv_cutoff'],
                max_interval=temp_param_dict['max_interval'], min_height=temp_param_dict['min_height'], min_peak=temp_param_dict['min_peak'],
                start=temp_param_dict['start'], end=temp_param_dict['end'], thresh_frac=temp_param_dict['thresh_frac'])
            #extract the spikes and make a dataframe for each sweep
            self.spike_df = {}
            self.rejected_spikes = {} if show_rejected else None
            for sweep in self.selected_sweeps:
                self.abf.setSweep(sweep)
                
                self.spike_df[sweep] = self.spike_extractor.process(self.abf.sweepX,self.abf.sweepY, self.abf.sweepC)
                if show_rejected:
                    self.rejected_spikes[sweep] = pd.DataFrame().from_dict(determine_rejected_spikes(self.spike_extractor, self.spike_df[sweep], self.abf.sweepY, self.abf.sweepX,
                    temp_param_dict)).T
                #self.rejected_spikes = None
                self.indiv_popup.setValue(sweep)
            #self.spike_df = pd.concat(self.spike_df)
        elif self.get_current_analysis() is 'subthres':
            self.spike_df = None
            self.rejected_spikes = None
            self.subthres_df, _ = analyze_subthres(self.abf, **self.subt_param_dict)
        self.indiv_popup.hide()
            
    
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
        print("Analysis changed")
        if hasattr(self, 'param_dict') is False:
            self.param_dict = {}
            old_params = {}
        else:
            old_params = copy.deepcopy(self.param_dict)
        self.get_analysis_params()
        #check if the parameters have changed
        if old_params != self.param_dict:
            print("Parameters changed - running analysis")
            #if they have changed, we need to run the analysis again
            self.analysis_changed_run()
        
        
    def analysis_changed_run(self):
        if self.abf is not None:
            self.run_indiv_analysis()
            self.plot_abf()
            self.refresh.setText("🔄 Refresh plot")

    
    def run_analysis(self):
        #check whether we are running spike or subthres
        if self.get_current_analysis() is 'spike':
            #run the folder analysis
            self.get_analysis_params()
            self.get_selected_protocol()
            
            df = self._inner_analysis_loop(self.selected_dir, self.param_dict,  self.selected_protocol)     
            save_data_frames(df[0], df[1], df[2], self.selected_dir, str(time.time())+self.outputTag.text(), self.bspikeFind.isChecked()
                             , self.brunningBin.isChecked(), self.brawData.isChecked())
            self.df = df[1]
        elif self.get_current_analysis() is 'subthres':
            self.get_analysis_params()
            self.get_selected_protocol()
            sweepwise_df, avg_df = self._inner_analysis_loop_subthres(self.selected_dir, self.subt_param_dict,  self.selected_protocol)
            save_subthres_data(avg_df, sweepwise_df, self.selected_dir, str(time.time())+self.outputTag.text())
            self.df = avg_df
        self.tableView.setModel(PandasModel(self.df, index='filename', parent=self.tableView))
        
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

    def create_plot(self, x, y, title=None, xlabel='', ylabel='', type='line', color='k', xlim=None, ylim=None, clear=False, raise_window=True):
        title = self._spawn_plot_window(name=title)
        #clear the figure
        if clear: 
            self.plot_windows[title]['figure'].figure.clear()
        try:
            self.axe1 = self.plot_windows[title]['figure'].figure.axes[0]
        except:
            self.axe1 = self.plot_windows[title]['figure'].figure.add_subplot(111)
        if type == 'line':
            self.axe1.plot(x, y, label=title)
        elif type == 'scatter':
            self.axe1.scatter(x, y, label=title)
        if xlim is not None:
            self.axe1.set_xlim(xlim)
        if ylim is not None:
            self.axe1.set_ylim(ylim)
        self.axe1.set_title(title)
        self.axe1.set_xlabel(xlabel)
        self.axe1.set_ylabel(ylabel)
        if raise_window:
            self.plot_windows[title]['window'].raise_()
        
    def _save_csv_for_current_file(self):
        #for the current abf run the analysis and save the csv
        self.run_indiv_analysis()
        if self.get_current_analysis() == 'spike':
            dfs = process_file(self.abf.abfFilePath, copy.deepcopy(self.param_dict), '')
            save_data_frames(dfs[1], dfs[0], dfs[2], self.selected_dir, str(time.time())+self.outputTag.text(), self.bspikeFind.isChecked(), self.brunningBin.isChecked(), self.brawData.isChecked())

    def _find_outliers(self, df):
        outlier_dect = IsolationForest(contamination='auto',random_state=42, n_jobs=-1)
        temp_df = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(df.select_dtypes(include=['float']))
        outlier_dect.fit(temp_df)
        labels = outlier_dect.predict(temp_df)
        return labels

    def _inner_analysis_loop(self, folder, param_dict, protocol_name):
        dfs = pd.DataFrame()
        df_spike_count = pd.DataFrame()
        df_running_avg_count = pd.DataFrame()
        filelist = glob.glob(folder + "/**/*.abf", recursive=True)
        popup = QProgressDialog("Running Analysis.", "Cancel", 0, len(filelist), None)
        popup.setWindowModality(QtCore.Qt.WindowModal)
        popup.forceShow()
        spike_count = []
        df_full = []
        df_running_avg = []
        parallel_processing = self.actionEnable_Parallel.isChecked()
        i = 0

        if parallel_processing:
            def update_iteration(_, i):
                popup.setValue(i)
                popup.setLabelText("Processing file " + str(i) + " of " + str(len(filelist)))
            pool = mp.Pool(mp.cpu_count())

            results = [pool.apply_async(process_file, args=(file, copy.deepcopy(param_dict), protocol_name), callback=partial(update_iteration, i=i)) for (i, file) in enumerate(filelist)]
            
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
                temp_df_spike_count, temp_full_df, temp_running_bin = process_file(f, copy.deepcopy(param_dict),protocol_name)
                spike_count.append(temp_df_spike_count)
                df_full.append(temp_full_df)
                df_running_avg.append(temp_running_bin)
        df_spike_count = pd.concat(spike_count, sort=True)
        dfs = pd.concat(df_full, sort=True)
        df_running_avg_count = pd.concat(df_running_avg, sort=False)
        popup.hide()
        #detect outliers
        df_spike_count['outlier'] = self._find_outliers(df_spike_count)
       

        return dfs, df_spike_count, df_running_avg_count

    def _inner_analysis_loop_subthres(self, folder, param_dict, protocol_name):
        filelist = glob.glob(folder + "/**/*.abf", recursive=True)
        popup = QProgressDialog("Operation in progress.", "Cancel", 0, len(filelist), None)
        popup.setWindowModality(QtCore.Qt.WindowModal)
        popup.forceShow()
        dfs = []
        for i, f in enumerate(filelist):
            popup.setValue(i)
            df = preprocess_abf_subthreshold(f, protocol_name, copy.deepcopy(param_dict))
            dfs.append(df)
        popup.hide()
        return pd.concat([x[0] for x in dfs], axis=0), pd.concat([x[1] for x in dfs], axis=0)


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
                #create an ISI x time plot
                isi = np.diff(self.spike_df[sweep]['peak_t'])
                
        #plot the rejected spikes if they exist
        if self.rejected_spikes is not None:
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
                
                row_dicts = [str(row.to_dict()) for index, row in self.rejected_spikes[sweep].iterrows()]
                #get the unique labels
                unique_labels = np.unique(row_dicts)
                for unique_type in unique_labels:
                    #rows that match the unique type
                    rows = self.rejected_spikes[sweep][self.rejected_spikes[sweep].apply(lambda x: str(x.to_dict()) == unique_type, axis=1)]
                    #plot the rejected spikes
                    self.axe1.scatter(self.abf.sweepX[rows.index.values], self.abf.sweepY[rows.index.values], color=color_dict[unique_type], s=10, zorder=99,)
                  
           
            #create a legend for the rejected spikes
            #create a list of patches for the legend
            patches = [mpatches.Patch(color=color_dict[label], label=label) for label in color_dict]
            #create the legend
            reject_spikes_legend = self.axe1.legend(handles=patches, loc='upper left', fontsize=8)

        #if the analysis was subthreshold, we need to plot the results
        if self.subthres_df is not None:
            lines = self.axe1.get_lines()
            cols = self.subthres_df.columns
            for sweep in self.selected_sweeps:
                self.abf.setSweep(sweep)
                real_sweep_number = sweepNumber_to_real_sweep_number(sweep)
                cols_for_sweep = [c for c in cols if real_sweep_number in c]
                if len(cols_for_sweep) == 0:
                    lines[sweep].set_color('#000000') #if no columns for the sweep, set the color to black
                    lines[sweep].set_alpha(0.1) #set the alpha to 0.1
                    continue
                temp_df = self.subthres_df[cols_for_sweep]
                #decay_fast, decay_slow, curve, r_squared_2p, r_squared_1p, p_decay = exp_decay_factor(dataT, dataV, dataI, time_after, abf_id=abf.name)
                #pull out the params, we want the decay, A1, b1, b2
                decay_fast = 1/temp_df[f"fast 2 phase tau {real_sweep_number}"].to_numpy()[0]
                decay_slow = 1/temp_df[f"slow 2 phase tau {real_sweep_number}"].to_numpy()[0]
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
                try:
                    sag_diff_plot = np.arange(avg_min, vm, 1)
                except:
                    sag_diff_plot = np.arange(avg_min, 0, 1, dtype=np.float64)
                self.axe1.scatter(dataT[min_point], dataV[min_point], c='r', marker='x', zorder=99, label="Min Point")
                self.axe1.scatter(dataT[end_index:upwardinfl], dataV[end_index:upwardinfl], c='g', zorder=99, label="Mean Vm Measured")
                self.axe1.plot(dataT[np.full(sag_diff_plot.shape[0], min_point, dtype=np.int64)], sag_diff_plot, label=f"Sag of {sag_diff}")

        #self.axe1.legend( bbox_to_anchor=(1.05, 1),
                       #  loc='upper left', borderaxespad=0.)
        #self.axe2.legend(loc='upper right')
        #self.axe1.add_artist(reject_spikes_legend)
        #create a span_selector for the time
        #if the span selector has been created, update the extents
        if hasattr(self, 'span'):
            self.span.new_axes(self.axe1)
            self.span.extents = (float(self.start_time.text()), float(self.end_time.text()))
        else:
            self.span = SpanSelector(self.axe1, self._mpl_span, 'horizontal', useblit=True,
                    props=dict(alpha=0.1, facecolor='red'), handle_props=dict(alpha=0.0), interactive=True)
            self.span.extents = (float(self.start_time.text()), float(self.end_time.text()))
        self.main_view.draw()

    def _mpl_span(self, min_value, max_value):
        #update the time values
        self.start_time.setValue(float(min_value))
        self.end_time.setValue(float(max_value))
        self.analysis_changed_run()

    def _plot_pyqtgraph(self):
        '''TODO'''
        self.main_view.clear()
        #self.main_view.figure.canvas.setFixedWidth(900)
        self.axe1 = self.main_view.addPlot(1,1, )
        self.axe2 = self.main_view.addPlot(2,1)
        self.axe1.addLegend()
        self.axe2.setXLink(self.axe1)
        self.get_selected_abf()
        self.get_selected_sweeps()
        #for the chosen sweeps
        if self.selected_sweeps == None:
            self.selected_sweeps = self.abf.sweepList

        #create a list of colors for the sweeps, in tab10
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

       
        for i, sweep in enumerate(self.selected_sweeps):
            self.abf.setSweep(sweep)
            self.axe1.plot(self.abf.sweepX, self.abf.sweepY, pen=pg.mkPen(colors[i], width=3), name="Sweep_" + str(sweep))
            #plot the dvdt
            self.axe2.plot(self.abf.sweepX[:-1], (np.diff(self.abf.sweepY)/np.diff(self.abf.sweepX))/1000,pen=pg.mkPen(colors[i]), name="Sweep_" + str(sweep))
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

    def _spawn_plot_window(self, name=None):
        #create a new plotting window when called, for plotting things like current x firing etc
        given_name = name if name is not None else f"Plot Window {len(self.plot_windows)+1}"
        #check if the window is already open
        if given_name in self.plot_windows:
            #if it is, show it
            #get the idx of the window
            idx = [x.windowTitle() for x in self.mdi.subWindowList()].index(given_name)
            self.mdi.subWindowList()[idx].showNormal()
            #also pull it to the front
            self.mdi.subWindowList()[idx].setFocus()
        else:
            #if it is not, create it, just a simple mdi window with a canvas in it
            #create a new plot window
            self.plot_windows[given_name] = {}
            self.plot_windows[given_name]['window'] = QtWidgets.QMdiSubWindow(self.mdi_area)
            self.plot_windows[given_name]['window'].setWindowTitle(given_name)

            self.plot_windows[given_name]['window'].setWidget(QtWidgets.QWidget())
            self.plot_windows[given_name]['window'].widget().setLayout(QtWidgets.QVBoxLayout())

            self.plot_windows[given_name]['figure'] = FigureCanvas(Figure(figsize=(15, 5)))
            self.plot_windows[given_name]['window'].widget().layout().addWidget(self.plot_windows[given_name]['figure'])

            self.plot_windows[given_name]['window'].setAttribute(Qt.WA_DeleteOnClose)

            self.plot_windows[given_name]['window'].show()
            self.plot_windows[given_name]['window'].setWindowModality(QtCore.Qt.WindowModal)
        return given_name

    def _run_script(self, checked=False, name=None):
        #try to spawn the script in the current terminal
        #if it fails, spawn a new terminal
        SCRIPT_PAIRS = {'actionOrganize_Abf': f'python {os.path.dirname(__file__)}/org_by_protocol.py', 
        'actionRun_APisolation': 'python run_APisolation.py', 'actionRun_APisolation_gui': 'python run_APisolation_gui.py'}
        try:
            print(f"Running {SCRIPT_PAIRS[name]}")
            os.system(SCRIPT_PAIRS[name])
        except:
            print(f"Failed to run {SCRIPT_PAIRS[name]}")
            
        return
    
    def _results_table_select(self, index):
        #get the selected row
        row = index.model()._dataframe.iloc[index.row()]
        #get the filename;
        filename = row['filename']+'.abf'
        #highlight that file in the file list
        for i in np.arange(self.file_list.count()):
            item = self.file_list.item(i)
            if item.text() == filename:
                item.setSelected(True)
                self.file_list.setCurrentRow(i)
                #fire a clicked event
                self.file_list.itemClicked.emit(item)
    
    def _prism_writer(self):
        #prismwritegui is a qwidget
        self.prismwritegui = PrismWriterGUI()
        self.prismwritegui.show()
        self.prismwritegui.raise_()

    ### window management functions
    def _view_window(self, action):
        window_list = [x.windowTitle() for x in self.mdi.subWindowList()]
        #check if the window is already open
        if action.text() in window_list:
            #if it is, show it
            #get the idx of the window
            idx = window_list.index(action.text())
            self.mdi.subWindowList()[idx].showNormal()
            #also pull it to the front
            self.mdi.subWindowList()[idx].setFocus()
        else:
            #if it is not, create it
            pass

    #close the view window 
    def _close_window(self, wind):
        #don't actually close the window, just hide it
        wind.hide()

    def exit(self):
        self.close()
        sys.exit()





class PandasModel(QAbstractTableModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe: pd.DataFrame, index=None, parent=None):
        QAbstractTableModel.__init__(self, parent)
        if index is not None:
            #clone the column to the index
            dataframe['_temp_index'] = dataframe[index].to_numpy()
            dataframe = dataframe.set_index('_temp_index')
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        """ Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])
        elif role == Qt.BackgroundColorRole and 'outlier' in self._dataframe.columns:
            if self._dataframe.iloc[index.row()]['outlier'] == -1:
                return QtGui.QColor(255, 0, 0)
        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole
    ):
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns.values[section])

            if orientation == Qt.Vertical:
                return str(self._dataframe.index.values[section])

        return None

    def sort(self, column: int, order=Qt.SortOrder):
        """Override method from QAbstractTableModel

        Sort the pandas DataFrame by column
        """
        self.layoutAboutToBeChanged.emit()
        self._dataframe = self._dataframe.sort_values(
            self._dataframe.columns[column], ascending=order == Qt.AscendingOrder
        )
        self.layoutChanged.emit()

def main():
    mp.freeze_support()
    app = QApplication([])
    widget = analysis_gui(app)
    widget.main_widget.show()
    #widget.children()[0].hide()
    #widget.children()[1].show()
    #pull focus to the ui
    #widget.children()[1].raise_()

    sys.exit(app.exec_())
