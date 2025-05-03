from . import databaseBuilderBase as dbb

from ..database import tsDatabase
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QTreeView, QVBoxLayout, QWidget, QFileSystemModel, QLabel, QLineEdit, QCommandLinkButton, QGroupBox, QTextEdit
from PySide2.QtGui import QStandardItem, QStandardItemModel
from PySide2.QtCore import QDir
import numpy as np
import sys
import time


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
        self.actionAddCell = self.menuFile.addAction('Add Cell')
        #link the button 

        self.database = tsDatabase.tsDatabase()

        self.folderLayout = QVBoxLayout()
        self.folderView = QTreeView(self.fileTreeFrame)
        self.folderView.setGeometry(10, 50, 780, 500)  # Adjust the size and position as needed
        
        self.folderView.setDragEnabled(True)
        self.folderView.setDragDropMode(QTreeView.DragOnly)
        self.folderView.setAcceptDrops(True)
        self.folderView.setDropIndicatorShown(True)

        self.folderLayout.addWidget(self.folderView)
        self.fileTreeFrame.setLayout(self.folderLayout)

        # Use the custom tree view for cellIndex
        self.cellIndexLayout = QVBoxLayout()
        self.cellIndex = CustomTreeView(parent=self.cellIndexFrame, update_callback=self._handleDropEvent)
        
        self.cellIndex.setGeometry(10, 50, 780, 500)  # Adjust the size and position as needed
        self.cellIndexFrame.setAcceptDrops(True)
        self.cellIndex.setDragDropMode(QTreeView.DropOnly)
        self.cellIndex.setDragEnabled(True)
        self.cellIndex.setDropIndicatorShown(True)

        self.cellIndexLayout.addWidget(self.cellIndex)
        self.cellIndexFrame.setLayout(self.cellIndexLayout)
        # Connect the open action to the openFolder method
        self.actionOpen.triggered.connect(self.openFolder)
        self.actionAddCell.triggered.connect(self._addCell)
        self.addCell.clicked.connect(self._addCell)
        self.addProtocol.clicked.connect(self._addProtocol)


        # Initialize the cell index model
        self.cellIndexModel = QStandardItemModel()
        self.cellIndexModel.setHorizontalHeaderLabels(['Cell Name', 'Recordings'])
        self.cellIndex.setModel(self.cellIndexModel)

        self.cellIndex.setAcceptDrops(True)
        self.cellIndex.setDragDropMode(QTreeView.DropOnly)

        self.cell_layout = None
        self.protocol_layout = None

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
        self._clearGroupBox()
        # Spawn some prompts in the GUI to add a cell.
        # We will add prompts to the group box.
        # Add a new row to the group box.
        if self.cell_layout is None:
            self.cell_layout = QVBoxLayout()
            self.cell_layout.addWidget(QLabel('Cell Name'))
            self.cellName = QLineEdit()
            self.cell_layout.addWidget(self.cellName)
            self.cell_notes = QTextEdit()
            self.cell_layout.addWidget(QLabel('Notes'))
            self.cell_layout.addWidget(self.cell_notes)
            confirm_button = QCommandLinkButton('Confirm')
            confirm_button.clicked.connect(self._addCellstoDatabase)
            self.cell_layout.addWidget(confirm_button)

        
        self.groupBox.setLayout(self.cell_layout)

    def _updateCell(self, event):
        # Implement the logic to update the cell(s)
        pass

    def _addProtocol(self):
        self._clearGroupBox()
        # Spawn some prompts in the GUI to add a protocol.
        # We will add prompts to the group box.
        # Add a new row to the group box.
        if self.protocol_layout is None:
            self.protocol_layout = QVBoxLayout()
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
        
        self.groupBox.setLayout(self.protocol_layout)
    
    def _addProtocolstoCell(self):
        # Add the protocols to the cell(s)
        cells = self.database.getCells()
        protocol_name = self.protocolName.text()
        protocol_description = self.protocolDescription.toPlainText()
        protocol_pharma = self.protocolPharma.text()
        protocol_temp = self.protocolTemp.text()
        for cell_name, recordings in cells.items():
            # Add the protocol to the cell
            self.database.addProtocol(cell_name, protocol_name, description=protocol_description, pharma=protocol_pharma, temp=protocol_temp)
        self._updateCellIndex()

        # Clear the protocol form fields
        self._clearGroupBox()

    def _addCellstoDatabase(self):
        # Add the cell(s) to the database
        cell_name = self.cellName.text()
        cell_notes = self.cell_notes.toPlainText()
        self.database.addEntry(cell_name)
        self._updateCellIndex()
        # Clear the cell form fields
        self._clearGroupBox()

    def _handleDropEvent(self, event):
        # Handle the drop event and update the cell index model
        mime_data = event.mimeData()
        drop_position = event.pos()  # Get the position of the drop event

        # Get the index of the item at the drop position
        index = self.cellIndex.indexAt(drop_position)
        if not index.isValid():
            # if the drop position is not created on an item, likely on an empty space
            # prompt the user to add a cell
            #create a popup to add a cell
            self._addCell()

        # Get the item at the drop position
        item = self.cellIndexModel.itemFromIndex(index)
        #if the user drops a file on a cell, add the file to the cell
        #get the type of the item
        item_type = item.hasChildren()
        # cheap way to check if the item is a cell
        # if the item is a cell, the item will have children
        if item_type:
            self._updateCellIndexWithFile(mime_data, item)
        else:
            #the item is not a cell, likely a protocol/recordings
            #get the parent of the item
            parent = item.parent()
            self._updateCellIndexWithFile(mime_data, parent, item)


    def _updateCellIndexWithFile(self, mime_data, cell=None, protocol=None):
        # Update the cell index model with the file path
        if cell is None:
            # we need to make a new cell,
            # this gets tricky because we need to check if the file is a protocol
            # TODO: Implement this
             #this will get tricky, we need to check the primary protocol and structure accordingly
            pass
        elif cell is not None and protocol is None:
            # Add the file to the protocol
            # we need to open the file to get the protocol name, etc. Luckily, we can use the tsDatabase class to do this
            file_dicts = {} #handle multiple files, this means the user can drop multiple files at once
            protocol_list = []
            if mime_data.hasUrls():
                for url in mime_data.urls():
                    url = url.toLocalFile()
                    file_dicts[url] = self.database.parseFile(url)
                    self.database.addProtocol(cell.text(), file_dicts[url]['protocol'], path=url)
            else:
                # I don't know what to do here, print a warning
                print('No file found')
        elif cell is not None and protocol is not None:
            #then the user is trying to add a recording to the protocol
            pass
        self._updateCellIndex()
        

    def _removeCell(self):
        pass 

    def _updateCellIndex(self):
        #update cell index treeview to display the nested data

        #get the expanded rows etc.
        expanded_rows = []
        for i in range(self.cellIndexModel.rowCount()):
            if self.cellIndex.isExpanded(self.cellIndexModel.index(i, 0)):
                expanded_rows.append(i)
    

        # Clear the existing model
        self.cellIndexModel.clear()
        self.cellIndexModel.setHorizontalHeaderLabels(['Cell Name', 'Recordings'])
        
        cells = self.database.getCells()

        for cell_name, recordings in cells.items():
            cell_item = QStandardItem(cell_name)
            for recording_type, recording in recordings.items():
                #check to make sure its not in the utility columns
                if (recording_type == "name" or recording_type == "sweep"):
                    continue
                #if recording is not None:  # Only add if there is a recording
                recording_type_item = QStandardItem(recording_type)
                recording_item = QStandardItem(recording)
                cell_item.appendRow([recording_type_item, recording_item])
            self.cellIndexModel.appendRow(cell_item)

        # Expand the rows that were previously expanded
        for row in expanded_rows:
            self.cellIndex.setExpanded(self.cellIndexModel.index(row, 0), True)

    def _clearGroupBox(self):
        def _innerclear():
            if self.groupBox.layout() is None:
                return True
            # Hide the layout
            #self.groupBox.layout().hide()
            self.groupBox.setLayout(None)
            return False
        _innerclear()
        
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

    def dragEnterEvent(self, event):
        event.acceptProposedAction()

    def dragMoveEvent(self, event):
        event.acceptProposedAction()
 
def run():
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = DatabaseBuilder()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())