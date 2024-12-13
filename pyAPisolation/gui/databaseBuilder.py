from . import databaseBuilderBase as dbb

from ..database import tsDatabase
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QTreeView, QVBoxLayout, QWidget, QFileSystemModel, QLabel, QLineEdit, QCommandLinkButton, QGroupBox, QTextEdit
from PySide2.QtGui import QStandardItem, QStandardItemModel
from PySide2.QtCore import QDir
import numpy as np
import sys


class DatabaseBuilder(dbb.Ui_databaseBuilderBase):
    def __init__(self):
        super(DatabaseBuilder, self).__init__()

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        # file should be the first menu in the menubar
        self.menuFile = self.menubar.children()[1]
        # add action to menuFile
        self.actionOpen = self.menuFile.addAction('Open Folder')
        self.actionSave = self.menuFile.addAction('Save')
        #add a seperator 
        self.actionAddCell = self.menuFile.addAction('Open Folder')
        #link the button 

        self.database = tsDatabase.tsDatabase()


        self.folderView = QTreeView(self.fileTreeFrame)
        self.folderView.setGeometry(10, 50, 780, 500)  # Adjust the size and position as needed

        # Use the custom tree view for cellIndex
        self.cellIndex = CustomTreeView(parent=self.cellIndexFrame, update_callback=self._handleDropEvent)
        self.cellIndex.setGeometry(10, 50, 780, 500)  # Adjust the size and position as needed


        # Connect the open action to the openFolder method
        self.actionOpen.triggered.connect(self.openFolder)
        self.actionAddCell.triggered.connect(self._addCell)
        self.addCell.clicked.connect(self._addCell)
        self.addProtocol.clicked.connect(self._addProtocol)


        # Initialize the cell index model
        self.cellIndexModel = QStandardItemModel()
        self.cellIndexModel.setHorizontalHeaderLabels(['Cell Name', 'Recordings'])
        self.cellIndex.setModel(self.cellIndexModel)

        self.folderView.setDragEnabled(True)
        self.folderView.setDragDropMode(QTreeView.DragOnly)
        self.cellIndex.setAcceptDrops(True)
        self.cellIndex.setDragDropMode(QTreeView.DropOnly)

    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)

    def openFolder(self):
        print('open folder')
        # open folder dialog
        folderPath = QFileDialog.getExistingDirectory(None, "Select Folder")

        if folderPath:
            # Set up the file system model
            self.model = QFileSystemModel()
            self.model.setRootPath(folderPath)

            # Set the name filters to only show .abf files
            self.model.setNameFilters(["*.abf"])
            self.model.setNameFilterDisables(False)

            # Set up the tree view
            self.folderView.setModel(self.model)
            self.folderView.setRootIndex(self.model.index(folderPath))
            self.folderView.setColumnHidden(1, True)  # Hide the size column
            self.folderView.setColumnHidden(2, True)  # Hide the type column       
            self.folderView.setDragEnabled(True)
            self.folderView.setDragDropMode(QTreeView.DragOnly)

    def _addCell(self):
        self.database.addEntry(str(np.random.rand(1)))
        self._updateCellIndex()

    def _updateCell(self, event):
        # Implement the logic to update the cell(s)
        pass

    def _addProtocol(self):
        # Spawn some prompts in the GUI to add a protocol.
        # We will add prompts to the group box.
        # Add a new row to the group box.
        self.protocol_layout = QVBoxLayout()
        self.groupBox.setLayout(self.protocol_layout)
        self.protocol_layout.addWidget(QLabel('Protocol Name'))
        self.protocolName = QLineEdit()
        self.protocol_layout.addWidget(self.protocolName)
        self.protocol_layout.addWidget(QLabel('Protocol Description'))
        self.protocolDescription = QTextEdit()
        self.protocol_layout.addWidget(self.protocolDescription)
        self.protocol_layout.addWidget(QLabel('Pharmacology'))
        self.protocolPharma = QLineEdit()
        self.protocol_layout.addWidget(self.protocolPharma)
        self.protocol_layout.addWidget(QLabel('Temperature'))
        self.protocolTemp = QLineEdit()
        self.protocol_layout.addWidget(self.protocolTemp)
        confirm_button = QCommandLinkButton('Confirm')
        confirm_button.clicked.connect(self._addProtocolstoCell)
        self.protocol_layout.addWidget(confirm_button)
    
    def _addProtocolstoCell(self):
        # Add the protocols to the cell(s)
        cells = self.database.getCells()
        protocol_name = self.protocolName.text()
        protocol_description = self.protocolDescription.toPlainText()
        protocol_pharma = self.protocolPharma.text()
        protocol_temp = self.protocolTemp.text()
        for cell_name, recordings in cells.items():
            # Add the protocol to the cell
            self.database.addProtocol(cell_name, protocol_name, protocol_description, protocol_pharma, protocol_temp)
        self._updateCellIndex()

    def _handleDropEvent(self, event):
        # Handle the drop event and update the cell index model
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            for url in mime_data.urls():
                file_path = url.toLocalFile()
                # Process the file path and update the cell index model
                self._updateCellIndexWithFile(file_path)

    def _updateCellIndexWithFile(self, file_path):
        # Implement the logic to update the cell index model with the dropped file
        cell_name = "New Cell"  # Example cell name, you can customize this
        recording_type = "Recording Type"  # Example recording type, you can customize this
        recording = file_path  # Use the file path as the recording

        cell_item = QStandardItem(cell_name)
        recording_type_item = QStandardItem(recording_type)
        recording_item = QStandardItem(recording)
        cell_item.appendRow([recording_type_item, recording_item])
        self.cellIndexModel.appendRow(cell_item)

    def _removeCell(self):
        pass 

    def _updateCellIndex(self):
        #update cell index treeview to display the nested data
        # Clear the existing model
        self.cellIndexModel.clear()
        self.cellIndexModel.setHorizontalHeaderLabels(['Cell Name', 'Recordings'])
        
        cells = self.database.getCells()

        for cell_name, recordings in cells.items():
            cell_item = QStandardItem(cell_name)
            for recording_type, recording in recordings.items():
                #check to make sure its not in the utility columns
                if (recording_type is "name" or recording_type is "sweep"):
                    continue
                if recording:  # Only add if there is a recording
                    recording_type_item = QStandardItem(recording_type)
                    recording_item = QStandardItem(recording)
                    cell_item.appendRow([recording_type_item, recording_item])
            self.cellIndexModel.appendRow(cell_item)


class CustomTreeView(QTreeView):
    def __init__(self, parent=None, update_callback=None):
        super(CustomTreeView, self).__init__(parent)
        self.update_callback = update_callback
        self.setDragEnabled(False)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QTreeView.DropOnly)

    def dropEvent(self, event):
        super(CustomTreeView, self).dropEvent(event)
        if self.update_callback:
            self.update_callback(event)
 
def run():
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = DatabaseBuilder()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())