from . import databaseBuilderBase as dbb
from PySide2.QtWidgets import QApplication
from PySide2.QtWidgets import QMainWindow
import sys


class DatabaseBuilder(dbb.databaseBuilderBase):
    def __init__(self):
        super(DatabaseBuilder, self).__init__()
    
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        #file should be the first menu in the menubar
        self.menuFile = self.menubar.children()[1]
        #add action to menuFile
        self.actionOpen = self.menuFile.addAction('Open Folder')
        self.actionSave = self.menuFile.addAction('Save')


def run():
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = DatabaseBuilder()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())