import ephyviewer
import numpy as np
from collections import OrderedDict
from ephyviewer import mkQApp, MainViewer, TraceViewer
from ephyviewer import InMemoryAnalogSignalSource
import pyabf
import glob
folder_loc = "C:\\Users\\SMest\\Downloads\\Data for Seminar (cluster)\\Marm-Parvo"
fileLlist = glob.glob(folder_loc + "\\*.abf")
ephyviewer.traceviewer.default_params[7]['value'] = 'w'
ephyviewer.traceviewer.default_by_channel_params[0]['value'] = 'k'

class fileListWidget(ephyviewer.QtGui.QListWidget):
    def __init__(self):
        super().__init__()
    def genList(self, fileList):
        self.addItems(fileList)

class controlsWidget(ephyviewer.QT.QBoxLayout):
    def __init__(self):
        super().__init__(ephyviewer.QtGui.QBoxLayout.LeftToRight)
        self.run_button = ephyviewer.QtGui.QPushButton("run")
        self.addWidget(self.run_button)
        self.name = 'analysis controls'

def clean_dicts(dict_v, keep):
    dict_return = OrderedDict()
    for key, value in dict_v.items():
        for x in keep:
            if x in key:
                dict_return.update({key:value})
                break
    return dict_return
    
class abfAnalyze():
    def __init__(self):
        self.app = ephyviewer.mkQApp()
        
        
        #Create the main window that can contain several viewers
        self.win = MainViewer(debug=True, show_auto_scale=True)
        self.fileList = fileListWidget()
        self.fileList.name = "Files"
        self.fileList.genList(fileLlist)
        self.fileList.itemClicked.connect(self.updateABF)
        dock = ephyviewer.QT.QDockWidget(self.fileList.name)
        dock.setObjectName(self.fileList.name)
        dock.setWidget(self.fileList)
        self.win.addDockWidget(ephyviewer.QT.LeftDockWidgetArea, dock)

    def run(self):
        
        self.win.show()
        self.app.exec_()

    def updateABF(self, item):
        
        filepath = item.text()
        self.abf = pyabf.ABF(filepath)
        self.sweepX = []
        self.sweepY = []
        for x in self.abf.sweepList:
            self.abf.setSweep(x)
            self.sweepX.append(self.abf.sweepX)
            self.sweepY.append(self.abf.sweepY)
        self.sweepX = np.vstack(self.sweepX)
        self.sweepY = np.vstack(self.sweepY)
        self.dvdt = np.diff(self.sweepY) / ((1/self.abf.dataRate)*1000)
        try:
            self.view1.closeEvent()
        except:
            pass
        self.view1 = TraceViewer.from_numpy(self.sweepY.T, self.abf.dataRate, 0, item.text())
        self.view2 = TraceViewer.from_numpy(self.dvdt.T, self.abf.dataRate, 0, item.text() + "dvdt")
        self.win.add_view(self.view1)
        self.win.add_view(self.view2)
        self.view1.auto_scale()
        self.view2.auto_scale()
        self.win.viewers = clean_dicts(self.win.viewers, [item.text()])
        self.view1.refresh()
        print(item.text())

    def clean_windows(self, keep):
        pass

    



def main():
    app = abfAnalyze()
    app.run()

main()