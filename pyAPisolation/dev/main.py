# This Python file uses the following encoding: utf-8
import sys
import os
import glob
import pyabf
import numpy as np
import pandas as pd



import scipy.signal as signal
from PySide2.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout
from PySide2.QtCore import QFile
from PySide2.QtUiTools import QUiLoader

from matplotlib.backends.qt_compat import QtWidgets
from matplotlib.backends.backend_qtagg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
print("Loaded external libraries")
from pyAPisolation.abf_featureextractor import folder_feature_extract, save_data_frames
from pyAPisolation.patch_utils import load_protocols

#import pyqtgraph as pg

import time
from ipfx.feature_extractor import SpikeFeatureExtractor

PLOT_BACKEND = 'matplotlib'


class analysis_gui(QWidget):
    def __init__(self):
        super(analysis_gui, self).__init__()
        self.load_ui()
        self.main_widget = self.children()[0]
        self.abf = None
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
        self.protocol_select.currentIndexChanged.connect(self.analysis_changed)
        self.bstim = self.main_widget.findChild(QWidget, "bstim")
        self.bessel = self.main_widget.findChild(QWidget, "bessel_filt")
        self.protocol_select.currentIndexChanged.connect(self.analysis_changed)
        self.bessel.textChanged.connect(self.analysis_changed)
        run_analysis = self.main_widget.findChild(QWidget, "run_analysis")
        run_analysis.clicked.connect(self.run_analysis)
        self.refresh = self.main_widget.findChild(QWidget, "refresh_plot")
        self.refresh.clicked.connect(self.analysis_changed_run)
        #link the settings buttons for the cm calc analysis
        

    
    def file_select(self):
        self.selected_dir = QFileDialog.getExistingDirectory()
        self.abf_list = glob.glob(self.selected_dir + "\\**\\*.abf", recursive=True)
        self.abf_list_name = [os.path.basename(x) for x in self.abf_list]
        self.pairs = [c for c in zip(self.abf_list_name, self.abf_list)]
        self.abf_file = self.pairs
        self.selected_sweeps = None
        #Generate the protocol list
        self.protocol_list = []
        for abf in self.abf_list:
            try:
                abf_obj = pyabf.ABF(abf, loadData=False)
                self.protocol_list.append(abf_obj.protocol)
            except:
                pass
        #we really only care about unique protocols
        #filter down to unique ones

        self.protocol_list = np.hstack(("[No Filter]", np.unique(self.protocol_list)))
        self.protocol_select.addItems(self.protocol_list)
        self.file_list.addItems(self.abf_list_name)

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
        b, a = signal.bessel(4, float(self.bessel.text()), 'low', norm='phase', fs=abf.dataRate)
        abf.data = signal.filtfilt(b, a, abf.data)
        return abf

    def run_indiv_analysis(self):
        #get the current abf
        self.abf = self.get_selected_abf()
        #get the current sweep(s)
        self.get_selected_sweeps()
        self.get_analysis_params()
        if self.selected_sweeps is None:
            self.selected_sweeps = self.abf.sweepList
        #create the spike extractor
        self.spike_extractor = SpikeFeatureExtractor(filter=0,  dv_cutoff=self.param_dict['dv_cutoff'],
         max_interval=self.param_dict['max_interval'], min_height=self.param_dict['min_height'], min_peak=self.param_dict['min_peak'])
        #extract the spikes and make a dataframe for each sweep
        self.spike_df = {}
        for sweep in self.selected_sweeps:
            self.abf.setSweep(sweep)
            self.spike_df[sweep] = self.spike_extractor.process(self.abf.sweepX,self.abf.sweepY, self.abf.sweepC)
        #self.spike_df = pd.concat(self.spike_df)
    
    
    def get_analysis_params(self):
        dv_cut = float(self.dvdt_thres.text())
        lowerlim = float(self.start_time.text())
        upperlim = float(self.end_time.text())
        tp_cut = float(self.thres_to_peak_time.text())
        min_cut = float(self.thres_to_peak_height.text())
        min_peak = float(self.min_peak_height.text())
        bstim_find = self.bstim.isChecked()
        bessel_filter = float(self.bessel.text())
        self.param_dict = {'filter': 0, 'dv_cutoff':dv_cut, 'start': lowerlim, 'end': upperlim, 'max_interval': tp_cut, 'min_height': min_cut, 'min_peak': min_peak, 
        'stim_find': bstim_find, 'bessel_filter': bessel_filter}
        return self.param_dict

    def analysis_changed(self):
        self.get_analysis_params()
        self.refresh
        
    def analysis_changed_run(self):
        if self.abf is not None:
            self.run_indiv_analysis( )
            self.plot_abf()

    def run_analysis(self):
        #run the folder analysis
        self.get_analysis_params()
        self.get_selected_protocol()
        df = folder_feature_extract(self.selected_dir, self.param_dict, False, self.selected_protocol)
        save_data_frames(df[0], df[1], df[2], self.selected_dir, str(time.time()))

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
        else:
            if (os.path.abspath(self.abf.abfFilePath) == os.path.abspath(self.selected_abf)) and (self.param_dict['bessel_filter'] <0):
                pass
            else:
                self.abf = self.filter_abf(pyabf.ABF(self.selected_abf))
        return self.abf
        
    def check_all(self):
        for sweep in self.checkbox_list:
            if isinstance(sweep, QtWidgets.QCheckBox):
                sweep.blockSignals()
                if sweep.isChecked():
                    sweep.setChecked(False)
                else:
                    sweep.setChecked(True)
                sweep.unblockSignals()
        self.plot_abf()

    def _plot_matplotlib(self):
        self.main_view.figure.clear()
        #self.main_view.figure.canvas.setFixedWidth(900)
        self.axe1 = self.main_view.figure.add_subplot(211)
        self.axe2 = self.main_view.figure.add_subplot(212, sharex=self.axe1)
        #self.main_view.figure.set_facecolor('#F0F0F0')
        #self.main_view.figure.set_edgecolor('#F0F0F0')
       # self.main_view.figure.set_dpi(100)
        #self.main_view.figure.set_tight_layout(True)
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
                    self.axe1.scatter(self.spike_df[sweep]['peak_t'], self.spike_df[sweep]['peak_v'], color='#FF0000', s=10, zorder=99, label='Spike Peak')
                    self.axe1.scatter(self.spike_df[sweep]['threshold_t'], self.spike_df[sweep]['threshold_v'], color='#00FF00', s=10, zorder=99, label='Threshold')
                    labeled_legend = True
                else:
                    self.axe1.scatter(self.spike_df[sweep]['peak_t'], self.spike_df[sweep]['peak_v'], color='#FF0000', s=10, zorder=99)
                    self.axe1.scatter(self.spike_df[sweep]['threshold_t'], self.spike_df[sweep]['threshold_v'], color='#00FF00', s=10, zorder=99)
        self.axe1.legend(loc='upper right')
        self.main_view.draw()

    def _plot_pyqtgraph(self):
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





if __name__ == "__main__":
    app = QApplication([])
    widget = analysis_gui()
    widget.show()
    sys.exit(app.exec_())
