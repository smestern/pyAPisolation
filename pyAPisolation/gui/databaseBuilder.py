from . import databaseBuilderBase as dbb

from ..database import tsDatabase
from PySide2.QtWidgets import QApplication, QMainWindow, QFileDialog, QTreeView, QVBoxLayout, QWidget, \
QFileSystemModel, QLabel, QLineEdit, QCommandLinkButton, QGroupBox, QTextEdit, QHeaderView, QAbstractItemView, \
QMenu, QAction, QWizard, QWizardPage, QHBoxLayout, QPushButton, \
QListWidget, QListWidgetItem, QProgressBar, QCheckBox, QSpinBox, \
QDoubleSpinBox, QComboBox, QFormLayout, QTableWidget, QTableWidgetItem
from PySide2.QtGui import QStandardItem, QStandardItemModel, QPalette
from PySide2.QtCore import QDir, Qt, QMimeData, QUrl, Signal
import numpy as np
import pandas as pd
import sys
import time
import glob
import os


class CustomFileSystemModel(QFileSystemModel):
    def __init__(self):
        super().__init__()
        self.protocol_data = {}  # Dictionary to store protocol names

    def columnCount(self, parent=None):
        return super().columnCount()  # No need to add +1 since we're modifying existing columns

    def data(self, index, role=Qt.DisplayRole):
        if index.column() == 0 and role == Qt.DisplayRole:  # Change to column 0 (filename column)
            file_path = self.filePath(index)
            protocol = self.protocol_data.get(file_path, "")['protocol'] if file_path in self.protocol_data else ""
            filename = super().data(index, role)
            return f"{filename} - {protocol}" if protocol else filename
        return super().data(index, role)

    def headerData(self, section, orientation, role):
        return super().headerData(section, orientation, role)

    def setProtocolData(self, file_path, protocol_name):
        index = self.filePath(self.index(file_path))
        self.protocol_data[index] = protocol_name
        self.dataChanged.emit(self.index(file_path), self.index(file_path))

class DatabaseBuilder(dbb.Ui_databaseBuilderBase):
    def __init__(self):
        super(DatabaseBuilder, self).__init__()

    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        
        # Add status bar for drag and drop feedback
        self.statusBar = MainWindow.statusBar()
        self.statusBar.showMessage("Ready - Drag .abf or .nwb files to add them to the database")
        
        # file should be the first menu in the menubar
        self.menuFile = self.menubar.children()[1]
        # add action to menuFile
        self.actionOpen = self.menuFile.addAction('Open Folder')
        self.actionLoad = self.menuFile.addAction('Load Database')
        self.actionSave = self.menuFile.addAction('Save Database')
        # add a seperator 
        self.menuFile.addSeparator()
        self.actionImportSpike = self.menuFile.addAction('Import Spike Data')
        self.menuFile.addSeparator()
        self.actionAddCell = self.menuFile.addAction('Add Cell')
        # link the button 

        self.database = tsDatabase.tsDatabase()

        self.folderLayout = QVBoxLayout()
        self.folderView = QTreeView(self.fileTreeFrame)
        self.folderView.setGeometry(10, 50, 780, 500)  # Adjust the size and position as needed
        
        # Enhanced drag settings will be set in openFolder method
        self.folderLayout.addWidget(self.folderView)
        self.fileTreeFrame.setLayout(self.folderLayout)

        # Use the custom tree view for cellIndex with enhanced drag and drop
        self.cellIndexLayout = QVBoxLayout()
        self.cellIndex = CustomTreeView(parent=self.cellIndexFrame, update_callback=self._handleDropEvent)
        
        self.cellIndex.setGeometry(10, 50, 780, 500)
        self.cellIndexFrame.setAcceptDrops(True)

        self.cellIndexLayout.addWidget(self.cellIndex)
        self.cellIndexFrame.setLayout(self.cellIndexLayout)
        
        # Connect the open action to the openFolder method
        self.actionOpen.triggered.connect(self.openFolder)
        self.actionLoad.triggered.connect(self.loadDatabase)
        self.actionSave.triggered.connect(self.saveDatabase)
        self.actionImportSpike.triggered.connect(self.importSpikeData)
        self.actionAddCell.triggered.connect(self._addCell)
        self.addCell.clicked.connect(self._addCell)
        self.addProtocol.clicked.connect(self._addProtocol)

        # Initialize the cell index model with more columns for spreadsheet-like view
        self.cellIndexModel = QStandardItemModel()
        self.cellIndexModel.setHorizontalHeaderLabels(['Cell Name', 'Protocol', 'Recording Path', 'Notes'])
        self.cellIndex.setModel(self.cellIndexModel)
        
        # Set initial column widths for better spreadsheet appearance
        self.cellIndex.setColumnWidth(0, 120)  # Cell Name
        self.cellIndex.setColumnWidth(1, 150)  # Protocol
        self.cellIndex.setColumnWidth(2, 300)  # Recording Path
        self.cellIndex.setColumnWidth(3, 200)  # Notes

        self.cell_layout = None
        self.protocol_layout = None

    def retranslateUi(self, MainWindow):
        super().retranslateUi(MainWindow)

    def openFolder(self):
        print('open folder')
        folderPath = QFileDialog.getExistingDirectory(None, "Select Folder")

        if folderPath:
            # Set up the custom file system model
            self.model = CustomFileSystemModel()
            self.model.setRootPath(folderPath)

            # Set the name filters to only show .abf files
            self.model.setNameFilters(["*.abf", "*.nwb"])
            self.model.setNameFilterDisables(False)

            # Set up the tree view with enhanced drag capabilities
            self.folderView.setModel(self.model)
            self.folderView.setRootIndex(self.model.index(folderPath))
            self.folderView.setColumnHidden(1, True)  # Hide the size column
            self.folderView.setColumnHidden(2, True)  # Hide the type column       
            
            # Enhanced drag and drop settings
            self.folderView.setDragEnabled(True)
            self.folderView.setDragDropMode(QTreeView.DragOnly)
            self.folderView.setDefaultDropAction(Qt.CopyAction)
            self.folderView.setSelectionMode(QTreeView.ExtendedSelection)  # Allow multi-select
            
            self.folderView.setColumnWidth(3, 200)  # Set the width of the protocol name column

            # Add styling for drag feedback
            self.folderView.setStyleSheet("""
                QTreeView {
                    selection-background-color: #3daee9;
                    selection-color: white;
                    alternate-background-color: #f9f9f9;
                }
                QTreeView::item:selected {
                    background-color: #3daee9;
                    color: white;
                    border: 1px solid #2e86ab;
                }
                QTreeView::item:hover {
                    background-color: #e8f4fd;
                }
            """)

            # Get protocol names and update the model
            file_list = glob.glob(folderPath + '/*.abf')
            file_list += glob.glob(folderPath + '/*.nwb')
            print(f'Found {len(file_list)} files in {folderPath}')
            for file in file_list:
                try:
                    protocol_name = self.database.parseFile(file)
                    self.model.setProtocolData(file, protocol_name)
                except Exception as e:
                    print(f"Error parsing file {file}: {e}")
            print('Protocol names updated in the file system model.')

    def saveDatabase(self):
        """Save the database to an Excel file"""
        from PySide2.QtWidgets import QFileDialog, QMessageBox
        
        # Open file dialog to get save location
        file_path, _ = QFileDialog.getSaveFileName(
            None,
            "Save Database",
            "database.xlsx",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            try:
                # Save the database using the tsDatabase save method
                saved_path = self.database.save(file_path)
                
                # Show success message
                QMessageBox.information(
                    None,
                    'Database Saved',
                    f'Database successfully saved to:\n{saved_path}\n\n'
                    f'The file contains multiple sheets:\n'
                    f'• CellIndex: Main cell and protocol data\n'
                    f'• Protocols: Protocol definitions\n'
                    f'• _cdb_config: Configuration data\n'
                    f'• _cdb_metadata: Database metadata'
                )
                
                # Update status bar
                self.statusBar.showMessage(f"Database saved to {saved_path}")
                
            except Exception as e:
                # Show error message
                QMessageBox.critical(
                    None,
                    'Save Error',
                    f'Failed to save database:\n{str(e)}'
                )
                print(f"Error saving database: {e}")

    def loadDatabase(self):
        """Load a database from an Excel file"""
        from PySide2.QtWidgets import QFileDialog, QMessageBox
        
        # Open file dialog to get file to load
        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "Load Database",
            "",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            try:
                # Load the database using the tsDatabase load method
                self.database.load_from_excel(file_path)
                
                # Update the cell index display
                self._updateCellIndex()
                
                # Show success message
                QMessageBox.information(
                    None,
                    'Database Loaded',
                    f'Database successfully loaded from:\n{file_path}'
                )
                
                # Update status bar
                self.statusBar.showMessage(f"Database loaded from {file_path}")
                
            except Exception as e:
                # Show error message
                QMessageBox.critical(
                    None,
                    'Load Error',
                    f'Failed to load database:\n{str(e)}'
                )
                print(f"Error loading database: {e}")

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

    def _handleDropEvent(self, event_info):
        """Enhanced drop event handler with better file path handling"""
        if hasattr(event_info, 'file_paths'):
            # New enhanced event info
            file_paths = event_info.file_paths
            target_index = event_info.target_index
            mime_data = event_info.mime_data
        else:
            # Legacy event handling
            mime_data = event_info.mimeData()
            target_index = self.cellIndex.indexAt(event_info.pos())
            file_paths = []
            if mime_data.hasUrls():
                for url in mime_data.urls():
                    file_path = url.toLocalFile()
                    if file_path and (file_path.endswith('.abf') or file_path.endswith('.nwb')):
                        file_paths.append(file_path)

        if not file_paths:
            print("No valid files to process")
            return

        # Determine target type and handle accordingly
        if not target_index.isValid():
            # Dropped in empty space - create new cell
            self._handleDropOnEmptySpace(file_paths)
        else:
            item = self.cellIndexModel.itemFromIndex(target_index)
            #get the items cell name, should be the first column of whatever row is dropped on
            temp_index = self.cellIndexModel.index(target_index.row(), 0, target_index.parent())
            cell_name = self.cellIndexModel.data(temp_index, Qt.DisplayRole)
    
            # Check if this is a top-level cell item (parent) or a protocol item (child)
            if item.parent() is None or cell_name != "":
                # This is a top-level cell item (no parent)
                if item.text() == cell_name:
                    # Handle dropping on a cell
                    self._handleDropOnCell(file_paths, item)
                else:
                    self._handleDropOnCell(file_paths, self.cellIndexModel.itemFromIndex(temp_index))
            else:
                # This is a child item (protocol/recording)
                #but we also need to figure out if its the protocol column or the recording column
                protocol_index = self.cellIndexModel.index(target_index.row(), 1, target_index.parent())
                recording_index = self.cellIndexModel.index(target_index.row(), 2, target_index.parent())
                parent_item = item.parent()
                self._handleDropOnProtocol(file_paths, parent_item, self.cellIndexModel.itemFromIndex(protocol_index))

    def _handleDropOnEmptySpace(self, file_paths):
        """Handle dropping files in empty space - prompt for new cell creation"""
        from PySide2.QtWidgets import QInputDialog, QMessageBox
        
        # Prompt user for cell name
        cell_name, ok = QInputDialog.getText(
            None, 
            'Create New Cell', 
            f'Enter name for new cell with {len(file_paths)} file(s):'
        )
        
        if ok and cell_name.strip():
            cell_name = cell_name.strip()
            
            # Add cell to database
            self.database.addEntry(cell_name)
            
            # Process each file and add to the cell
            protocols_added = set()
            for file_path in file_paths:
                try:
                    file_info = self.database.parseFile(file_path)
                    protocol_name = file_info.get('protocol', 'Unknown')
                    
                    # Add protocol if not already added
                    if protocol_name not in protocols_added:
                        self.database.addProtocol(cell_name, protocol_name, path=file_path)
                        protocols_added.add(protocol_name)
                    else:
                        # Update existing protocol with additional recording
                        self.database.updateEntry(cell_name, **{protocol_name: file_path})
                        
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")
            
            # Update the display
            self._updateCellIndex()
            
            # Show success message
            QMessageBox.information(
                None, 
                'Success', 
                f'Created cell "{cell_name}" with {len(file_paths)} file(s) across {len(protocols_added)} protocol(s)'
            )

    def _handleDropOnCell(self, file_paths, cell_item):
        """Handle dropping files on an existing cell"""
        cell_name = cell_item.text()
        
        protocols_added = set()
        protocols_updated = set()
        
        for file_path in file_paths:
            try:
                file_info = self.database.parseFile(file_path)
                protocol_name = file_info.get('protocol', 'Unknown')
                
                # Check if protocol already exists for this cell
                cells = self.database.getCells()
                cell_data = cells.get(cell_name, {})
                
                if protocol_name in cell_data:
                    # Protocol exists, update it
                    self.database.updateEntry(cell_name, **{protocol_name: file_path})
                    protocols_updated.add(protocol_name)
                else:
                    # New protocol for this cell
                    self.database.addProtocol(cell_name, protocol_name, path=file_path)
                    protocols_added.add(protocol_name)
                    
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        # Update the display
        self._updateCellIndex()
        
        # Show feedback
        message_parts = []
        if protocols_added:
            message_parts.append(f"Added {len(protocols_added)} new protocol(s)")
        if protocols_updated:
            message_parts.append(f"Updated {len(protocols_updated)} existing protocol(s)")
        
        if message_parts:
            from PySide2.QtWidgets import QMessageBox
            QMessageBox.information(
                None, 
                'Files Added', 
                f'Cell "{cell_name}": {", ".join(message_parts)}'
            )

    def _handleDropOnProtocol(self, file_paths, cell_item, protocol_item):
        """Handle dropping files on a specific protocol"""
        cell_name = cell_item.text()
        protocol_name = protocol_item.text()
        
        files_added = 0
        for file_path in file_paths:
            try:
                # Verify the file matches the protocol
                file_info = self.database.parseFile(file_path)
                file_protocol = file_info.get('protocol', 'Unknown')
                
                if file_protocol == protocol_name:
                    # File matches protocol, add it
                    self.database.updateEntry(cell_name, **{protocol_name: file_path})
                    files_added += 1
                else:
                    # File doesn't match, could offer to add as new protocol
                    from PySide2.QtWidgets import QMessageBox
                    reply = QMessageBox.question(
                        None,
                        'Protocol Mismatch',
                        f'File protocol "{file_protocol}" doesn\'t match target protocol "{protocol_name}".\n'
                        f'Add as new protocol instead?',
                        QMessageBox.Yes | QMessageBox.No
                    )
                    
                    if reply == QMessageBox.Yes:
                        self.database.addProtocol(cell_name, file_protocol, path=file_path)
                        files_added += 1
                    else:
                        #add it not as a protocol but as 
                        self.database.updateEntry(cell_name, **{protocol_name: file_path})
                        
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
        
        # Update the display
        self._updateCellIndex()
        
        # Show feedback
        if files_added > 0:
            from PySide2.QtWidgets import QMessageBox
            QMessageBox.information(
                None,
                'Files Added',
                f'Added {files_added} file(s) to protocol "{protocol_name}" in cell "{cell_name}"'
            )

    def _updateCellIndex(self):
        """Update cell index treeview to display data in spreadsheet format"""
        
        # Get the expanded rows etc.
        expanded_rows = []
        for i in range(self.cellIndexModel.rowCount()):
            if self.cellIndex.isExpanded(self.cellIndexModel.index(i, 0)):
                expanded_rows.append(i)
    
        # Clear the existing model
        self.cellIndexModel.clear()
        self.cellIndexModel.setHorizontalHeaderLabels([
            'Cell Name', 'Protocol', 'Recording Path', 'Notes'
        ])
        
        cells = self.database.getCells()

        for cell_name, recordings in cells.items():
            cell_item = QStandardItem(cell_name)
            cell_protocol = QStandardItem("")  # Empty protocol for parent
            cell_path = QStandardItem("")      # Empty path for parent
            cell_notes = QStandardItem("")     # Notes could be added later
            
            # Make the cell name row editable
            cell_item.setEditable(True)
            cell_notes.setEditable(True)
            
            for recording_type, recording in recordings.items():
                # Check to make sure its not in the utility columns
                if recording_type in ["name", "sweep"]:
                    continue
                    
                # Create child items for each recording
                protocol_item = QStandardItem(recording_type)
                path_item = QStandardItem(str(recording) if recording else "")
                notes_item = QStandardItem("")
                empty_cell = QStandardItem("")  # For cell name column
                
                # Make items editable
                protocol_item.setEditable(True)
                path_item.setEditable(True)
                notes_item.setEditable(True)
                
                cell_item.appendRow([empty_cell, protocol_item, path_item, notes_item])
                
            self.cellIndexModel.appendRow([cell_item, cell_protocol, cell_path, cell_notes])

        # Expand the rows that were previously expanded
        for row in expanded_rows:
            self.cellIndex.setExpanded(self.cellIndexModel.index(row, 0), True)
            
        # Resize columns to content after update
        #self.cellIndex.resizeColumnsToContents()
        for row in expanded_rows:
            self.cellIndex.setExpanded(self.cellIndexModel.index(row, 0), True)

    def _clearGroupBox(self):
        def _innerclear():
            if self.groupBox.layout() is None:
                return True
            # Hide the layout
            for i in range(self.groupBox.layout().count()):
                widget = self.groupBox.layout().itemAt(i).widget()
                if widget is not None:
                    widget.hide()
                    widget.deleteLater()
            self.groupBox.layout().deleteLater()
            self.groupBox.setLayout(None)
            self.cell_layout = None
            self.protocol_layout = None
            return False
        _innerclear()

    def importSpikeData(self):
        """Launch a wizard to import spike analysis data from CSV files"""
        wizard = SpikeDataImportWizard(self.database, self)
        if wizard.exec_() == wizard.Accepted:
            # Update the cell index display after successful import
            self._updateCellIndex()
            self.statusBar.showMessage("Spike data import completed successfully")
        
class CustomTreeView(QTreeView):
    def __init__(self, parent=None, update_callback=None):
        super(CustomTreeView, self).__init__(parent)
        self.update_callback = update_callback
        
        # Enhanced drag and drop settings
        self.setDragEnabled(False)  # We don't want to drag from this view
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QTreeView.DropOnly)
        self.setDefaultDropAction(Qt.CopyAction)
        
        # Make it more spreadsheet-like
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.setItemsExpandable(True)
        self.setRootIsDecorated(True)
        
        # Configure the header
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setDefaultSectionSize(150)
        header.setMinimumSectionSize(80)
        
        # Enhanced styling with drop indicators
        self.setStyleSheet("""
            QTreeView {
                gridline-color: #d0d0d0;
                alternate-background-color: #f9f9f9;
                background-color: white;
                selection-background-color: #3daee9;
                selection-color: white;
                show-decoration-selected: 1;
            }
            QTreeView::item {
                border-right: 1px solid #d0d0d0;
                padding: 4px;
                min-height: 20px;
            }
            QTreeView::item:selected {
                background-color: #3daee9;
                color: white;
            }
            QTreeView::item:hover {
                background-color: #e8f4fd;
            }
            QTreeView::drop-indicator {
                background-color: #ff6b6b;
                height: 3px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                padding: 4px;
                font-weight: bold;
            }
        """)
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)
        
        # Enable keyboard navigation
        self.setTabKeyNavigation(True)
        
    def showContextMenu(self, position):
        """Show context menu with spreadsheet-like options"""
        index = self.indexAt(position)
        menu = QMenu(self)
        
        if index.isValid():
            # Add common spreadsheet actions
            edit_action = QAction("Edit Cell", self)
            edit_action.triggered.connect(lambda: self.edit(index))
            menu.addAction(edit_action)
            
            clear_action = QAction("Clear Cell", self)
            clear_action.triggered.connect(lambda: self.clearCell(index))
            menu.addAction(clear_action)
            
            menu.addSeparator()
            
            copy_action = QAction("Copy", self)
            copy_action.triggered.connect(self.copySelection)
            menu.addAction(copy_action)
            
            paste_action = QAction("Paste", self)
            paste_action.triggered.connect(self.pasteSelection)
            menu.addAction(paste_action)
            
        menu.addSeparator()
        
        resize_action = QAction("Resize Columns to Contents", self)
        resize_action.triggered.connect(self.resizeColumnsToContents)
        menu.addAction(resize_action)
        
        menu.exec_(self.mapToGlobal(position))
        
    def clearCell(self, index):
        """Clear the content of a cell"""
        if index.isValid() and index.flags() & Qt.ItemIsEditable:
            self.model().setData(index, "", Qt.EditRole)
            
    def copySelection(self):
        """Copy selected cell content to clipboard"""
        selection = self.selectionModel().selectedIndexes()
        if selection:
            app = QApplication.instance()
            clipboard = app.clipboard()
            clipboard.setText(selection[0].data(Qt.DisplayRole) or "")
            
    def pasteSelection(self):
        """Paste clipboard content to current cell"""
        current = self.currentIndex()
        if current.isValid() and current.flags() & Qt.ItemIsEditable:
            app = QApplication.instance()
            clipboard = app.clipboard()
            text = clipboard.text()
            self.model().setData(current, text, Qt.EditRole)
        
    def keyPressEvent(self, event):
        """Enhanced keyboard navigation for spreadsheet-like behavior"""
        if event.key() == Qt.Key_Tab:
            # Move to next column
            current = self.currentIndex()
            if current.isValid():
                next_column = current.sibling(current.row(), current.column() + 1)
                if next_column.isValid():
                    self.setCurrentIndex(next_column)
                else:
                    # Move to first column of next row
                    next_row = current.sibling(current.row() + 1, 0)
                    if next_row.isValid():
                        self.setCurrentIndex(next_row)
            event.accept()
            return
        elif event.key() == Qt.Key_Backtab:
            # Move to previous column
            current = self.currentIndex()
            if current.isValid():
                prev_column = current.sibling(current.row(), current.column() - 1)
                if prev_column.isValid():
                    self.setCurrentIndex(prev_column)
                else:
                    # Move to last column of previous row
                    prev_row = current.sibling(current.row() - 1, self.model().columnCount() - 1)
                    if prev_row.isValid():
                        self.setCurrentIndex(prev_row)
            event.accept()
            return
        elif event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            # Move to next row, same column
            current = self.currentIndex()
            if current.isValid():
                next_row = current.sibling(current.row() + 1, current.column())
                if next_row.isValid():
                    self.setCurrentIndex(next_row)
            event.accept()
            return
        elif event.key() == Qt.Key_F2:
            # Start editing current cell
            current = self.currentIndex()
            if current.isValid():
                self.edit(current)
            event.accept()
            return
        elif event.key() == Qt.Key_Delete:
            # Clear current cell content
            current = self.currentIndex()
            if current.isValid() and current.flags() & Qt.ItemIsEditable:
                self.model().setData(current, "", Qt.EditRole)
            event.accept()
            return
        
        # For arrow keys, ensure single cell selection
        if event.key() in [Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right]:
            super().keyPressEvent(event)
            # Ensure only one cell is selected
            current = self.currentIndex()
            if current.isValid():
                self.selectionModel().clearSelection()
                self.selectionModel().select(current, self.selectionModel().Select)
            return
            
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Enhanced mouse behavior for spreadsheet-like cell selection"""
        super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            index = self.indexAt(event.pos())
            if index.isValid():
                # Clear selection and select only the clicked cell
                self.selectionModel().clearSelection()
                self.selectionModel().select(index, self.selectionModel().Select)
                self.setCurrentIndex(index)

    def dragEnterEvent(self, event):
        """Handle drag enter events for files"""
        if event.mimeData().hasUrls():
            # Check if any of the URLs are valid file paths
            valid_files = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path and (file_path.endswith('.abf') or file_path.endswith('.nwb')):
                    valid_files.append(file_path)
            
            if valid_files:
                event.acceptProposedAction()
                self.setStyleSheet(self.styleSheet() + """
                    QTreeView {
                        border: 2px dashed #3daee9;
                        background-color: #f0f8ff;
                    }
                """)
            else:
                event.ignore()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Handle drag move events to show drop indicators"""
        if event.mimeData().hasUrls():
            # Get the index under the cursor
            index = self.indexAt(event.pos())
            
            if index.isValid():
                # Highlight the target item
                self.setCurrentIndex(index)
                event.acceptProposedAction()
            else:
                # Allow dropping in empty space to create new cells
                event.acceptProposedAction()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        """Reset styling when drag leaves the widget"""
        # Reset the styling
        self.setStyleSheet("""
            QTreeView {
                gridline-color: #d0d0d0;
                alternate-background-color: #f9f9f9;
                background-color: white;
                selection-background-color: #3daee9;
                selection-color: white;
                show-decoration-selected: 1;
            }
            QTreeView::item {
                border-right: 1px solid #d0d0d0;
                padding: 4px;
                min-height: 20px;
            }
            QTreeView::item:selected {
                background-color: #3daee9;
                color: white;
            }
            QTreeView::item:hover {
                background-color: #e8f4fd;
            }
            QTreeView::drop-indicator {
                background-color: #ff6b6b;
                height: 3px;
            }
            QHeaderView::section {
                background-color: #f0f0f0;
                border: 1px solid #d0d0d0;
                padding: 4px;
                font-weight: bold;
            }
        """ )

    def dropEvent(self, event):
        """Enhanced drop event handling"""
        # Reset styling first
        self.dragLeaveEvent(event)
        
        if event.mimeData().hasUrls():
            # Get drop position and target
            drop_position = event.pos()
            target_index = self.indexAt(drop_position)
            
            # Extract file paths
            file_paths = []
            for url in event.mimeData().urls():
                file_path = url.toLocalFile()
                if file_path and (file_path.endswith('.abf') or file_path.endswith('.nwb')):
                    file_paths.append(file_path)
            
            if file_paths:
                # Show visual feedback
                self._showDropFeedback(len(file_paths), target_index)
                
                # Call the update callback with enhanced information
                if self.update_callback:
                    enhanced_event = DropEventInfo(
                        mime_data=event.mimeData(),
                        target_index=target_index,
                        file_paths=file_paths,
                        drop_position=drop_position
                    )
                    self.update_callback(enhanced_event)
                
                event.acceptProposedAction()
            else:
                event.ignore()
        else:
            event.ignore()

    def _showDropFeedback(self, file_count, target_index):
        """Show visual feedback for successful drop"""
        from PySide2.QtWidgets import QMessageBox
        
        if target_index.isValid():
            item = self.model().itemFromIndex(target_index)
            if item and item.hasChildren():
                # Dropped on a cell
                message = f"Added {file_count} file(s) to cell: {item.text()}"
            else:
                # Dropped on a protocol/recording
                parent = item.parent() if item else None
                if parent:
                    message = f"Added {file_count} file(s) to protocol in cell: {parent.text()}"
                else:
                    message = f"Added {file_count} file(s)"
        else:
            # Dropped in empty space
            message = f"Ready to create new cell with {file_count} file(s)"
        
        # You could replace this with a status bar message or tooltip
        print(f"Drop feedback: {message}")

# Add a helper class for enhanced drop event information
class DropEventInfo:
    def __init__(self, mime_data, target_index, file_paths, drop_position):
        self.mime_data = mime_data
        self.target_index = target_index
        self.file_paths = file_paths
        self.drop_position = drop_position
        self.pos = lambda: drop_position  # For compatibility with existing code

    def mimeData(self):
        return self.mime_data


class SpikeDataImportWizard(QWizard):
    """Wizard for importing spike analysis data"""
    
    # Page IDs
    PAGE_INTRO = 0
    PAGE_FILE_SELECT = 1
    PAGE_PREVIEW = 2
    PAGE_MAPPING = 3
    PAGE_OPTIONS = 4
    PAGE_IMPORT = 5
    PAGE_COMPLETE = 6
    
    def __init__(self, database, parent=None):
        super().__init__(parent.centralwidget)
        self.database = database
        self.csv_files = []
        self.selected_files = []
        self.preview_data = {}
        self.column_mappings = {}
        self.import_options = {}
        
        self.setWindowTitle("Spike Data Import Wizard")
        self.setWizardStyle(QWizard.ModernStyle)
        self.setFixedSize(800, 600)
        
        # Add pages
        self.addPage(IntroPage())
        self.addPage(FileSelectionPage())
        self.addPage(PreviewPage())
        self.addPage(ColumnMappingPage())
        self.addPage(ImportOptionsPage())
        self.addPage(ImportProgressPage())
        self.addPage(CompletePage())

class IntroPage(QWizardPage):
    """Introduction page explaining the import process"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Spike Data Import Wizard")
        self.setSubTitle("Import spike analysis data from CSV files into your database")
        
        layout = QVBoxLayout()
        
        intro_text = QLabel("""
        <h3>Welcome to the Spike Data Import Wizard</h3>
        
        <p>This wizard will help you import spike analysis data from CSV files (like spike_count_.csv) 
        into your cell database. The process involves the following steps:</p>
        
        <ol>
        <li><b>File Selection:</b> Choose the CSV files containing spike analysis data</li>
        <li><b>Data Preview:</b> Review the structure and content of your data</li>
        <li><b>Column Mapping:</b> Map CSV columns to database fields</li>
        <li><b>Import Options:</b> Configure how the data should be imported</li>
        <li><b>Import Process:</b> Import the data into your database</li>
        </ol>
        
        <p><b>Supported file formats:</b> CSV files with spike analysis results</p>
        <p><b>Expected data:</b> Files should contain recording paths, spike counts, and analysis metrics</p>
        
        <p>Click <b>Next</b> to begin the import process.</p>
        """)
        intro_text.setWordWrap(True)
        layout.addWidget(intro_text)
        
        self.setLayout(layout)

class FileSelectionPage(QWizardPage):
    """Page for selecting CSV files to import"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Select CSV Files")
        self.setSubTitle("Choose the CSV files containing spike analysis data")
        
        layout = QVBoxLayout()
        
        # Instructions
        instructions = QLabel("Select one or more CSV files containing spike analysis data:")
        layout.addWidget(instructions)
        
        # File selection buttons
        button_layout = QHBoxLayout()
        self.browse_button = QPushButton("Browse for Files...")
        self.browse_button.clicked.connect(self.browseFiles)
        button_layout.addWidget(self.browse_button)
        
        self.browse_folder_button = QPushButton("Browse Folder...")
        self.browse_folder_button.clicked.connect(self.browseFolder)
        button_layout.addWidget(self.browse_folder_button)
        
        button_layout.addStretch()
        layout.addLayout(button_layout)
        
        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        layout.addWidget(self.file_list)
        
        # Remove button
        self.remove_button = QPushButton("Remove Selected")
        self.remove_button.clicked.connect(self.removeSelected)
        self.remove_button.setEnabled(False)
        layout.addWidget(self.remove_button)
        
        self.file_list.itemSelectionChanged.connect(self.updateRemoveButton)
        self.file_list.item()
        
        self.setLayout(layout)
        
    def browseFiles(self):
        """Browse for individual CSV or XLSX files"""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Data Files",
            "",
            "Data Files (*.csv *.xlsx);;CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        if files:
            wizard = self.wizard()
            for file_path in files:
                if file_path not in wizard.csv_files:
                    wizard.csv_files.append(file_path)
                    item = QListWidgetItem(os.path.basename(file_path))
                    item.setData(Qt.UserRole, file_path)
                    item.setToolTip(file_path)
                    self.file_list.addItem(item)
            
    def browseFolder(self):
        """Browse for a folder containing CSV or XLSX files"""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Data Files")
        if folder:
            data_files = []
            for file in os.listdir(folder):
                if file.lower().endswith('.csv') or file.lower().endswith('.xlsx'):
                    data_files.append(os.path.join(folder, file))
            if data_files:
                wizard = self.wizard()
                for file_path in data_files:
                    if file_path not in wizard.csv_files:
                        wizard.csv_files.append(file_path)
                        item = QListWidgetItem(os.path.basename(file_path))
                        item.setData(Qt.UserRole, file_path)
                        item.setToolTip(file_path)
                        self.file_list.addItem(item)
            else:
                QMessageBox.information(self, "No Data Files", "No CSV or XLSX files found in the selected folder.")
    
    def removeSelected(self):
        """Remove selected files from the list"""
        wizard = self.wizard()
        for item in self.file_list.selectedItems():
            file_path = item.data(Qt.UserRole)
            if file_path in wizard.csv_files:
                wizard.csv_files.remove(file_path)
            self.file_list.takeItem(self.file_list.row(item))
    
    def updateRemoveButton(self):
        """Enable/disable remove button based on selection"""
        self.remove_button.setEnabled(len(self.file_list.selectedItems()) > 0)
    
    def isComplete(self):
        """Page is complete when at least one file is selected"""
        return len(self.wizard().csv_files) > 0

class PreviewPage(QWizardPage):
    """Page for previewing the selected CSV files"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Data Preview")
        self.setSubTitle("Review the structure and content of your CSV files")
        
        layout = QVBoxLayout()
        
        # File selector
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("Preview file:"))
        self.file_combo = QComboBox()
        self.file_combo.currentTextChanged.connect(self.updatePreview)
        file_layout.addWidget(self.file_combo)
        file_layout.addStretch()
        layout.addLayout(file_layout)
        
        # Preview table
        self.preview_table = QTableWidget()
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        layout.addWidget(self.preview_table)
        
        # Info labels
        self.info_label = QLabel()
        layout.addWidget(self.info_label)
        
        self.setLayout(layout)
    
    def initializePage(self):
        """Initialize the page when entering"""
        wizard = self.wizard()
        # Clear and populate file combo
        self.file_combo.clear()
        for file_path in wizard.csv_files:
            self.file_combo.addItem(os.path.basename(file_path), file_path)
        # Load preview data for all files
        wizard.preview_data = {}
        for file_path in wizard.csv_files:
            try:
                if file_path.lower().endswith('.csv'):
                    df = pd.read_csv(file_path, nrows=10)
                elif file_path.lower().endswith('.xlsx'):
                    df = pd.read_excel(file_path, nrows=10)
                else:
                    continue
                wizard.preview_data[file_path] = df
            except Exception as e:
                QMessageBox.warning(self, "Preview Error", 
                                  f"Could not preview file {os.path.basename(file_path)}:\n{str(e)}")
        # Update preview for first file
        if wizard.csv_files:
            self.updatePreview()
    
    def updatePreview(self):
        """Update the preview table for the selected file"""
        current_file = self.file_combo.currentData()
        if not current_file:
            return
        
        wizard = self.wizard()
        if current_file in wizard.preview_data:
            df = wizard.preview_data[current_file]
            
            # Update table
            self.preview_table.setRowCount(len(df))
            self.preview_table.setColumnCount(len(df.columns))
            self.preview_table.setHorizontalHeaderLabels(df.columns.tolist())
            
            for row in range(len(df)):
                for col in range(len(df.columns)):
                    item = QTableWidgetItem(str(df.iloc[row, col]))
                    self.preview_table.setItem(row, col, item)
            
            # Resize columns to content
            self.preview_table.resizeColumnsToContents()
            
            # Update info label
            try:
                full_df = pd.read_csv(current_file)
                self.info_label.setText(f"File: {os.path.basename(current_file)} | "
                                      f"Rows: {len(full_df)} | "
                                      f"Columns: {len(full_df.columns)} | "
                                      f"Showing first 10 rows")
            except:
                self.info_label.setText(f"File: {os.path.basename(current_file)} | "
                                      f"Showing first 10 rows")

class ColumnMappingPage(QWizardPage):
    """Page for mapping CSV columns to database fields"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Column Mapping")
        self.setSubTitle("Map CSV columns to database fields")
        
        layout = QVBoxLayout()
        
        instructions = QLabel("""
        Map the columns from your CSV files to the appropriate database fields.
        The wizard will attempt to auto-detect common column mappings based on column names.
        """)
        instructions.setWordWrap(True)
        layout.addWidget(instructions)
        
        # Mapping form
        self.mapping_form = QFormLayout()
        
        # Common field mappings
        self.field_combos = {}
        common_fields = [
            ("Recording Path", "path to the recording file"),
            ("Cell Name", "name or identifier of the cell"),
            ("Protocol", "protocol or experiment type"),
            ("Spike Count", "number of spikes detected"),
            ("Spike Rate", "firing rate (Hz)"),
            ("ISI Mean", "mean inter-spike interval"),
            ("ISI CV", "coefficient of variation of ISI"),
            ("First Spike Latency", "latency to first spike"),
            ("Sweep Number", "sweep or trial number")
        ]
        
        for field_name, description in common_fields:
            combo = QComboBox()
            combo.addItem("-- Not Mapped --", None)
            self.field_combos[field_name] = combo
            self.mapping_form.addRow(f"{field_name}:", combo)
        
        layout.addLayout(self.mapping_form)
        
        # Auto-detect button
        self.auto_detect_button = QPushButton("Auto-Detect Mappings")
        self.auto_detect_button.clicked.connect(self.autoDetectMappings)
        layout.addWidget(self.auto_detect_button)
        
        self.setLayout(layout)
    
    def initializePage(self):
        """Initialize the page when entering"""
        wizard = self.wizard()
        
        # Get all unique columns from all files
        all_columns = set()
        for file_path in wizard.csv_files:
            if file_path in wizard.preview_data:
                df = wizard.preview_data[file_path]
                all_columns.update(df.columns.tolist())
        
        # Populate combo boxes with available columns
        for combo in self.field_combos.values():
            combo.clear()
            combo.addItem("-- Not Mapped --", None)
            for column in sorted(all_columns):
                combo.addItem(column, column)
        
        # Auto-detect mappings
        self.autoDetectMappings()
    
    def autoDetectMappings(self):
        """Auto-detect column mappings based on common patterns"""
        wizard = self.wizard()
        
        # Get all columns
        all_columns = set()
        for file_path in wizard.csv_files:
            if file_path in wizard.preview_data:
                df = wizard.preview_data[file_path]
                all_columns.update(df.columns.tolist())
        
        # Mapping patterns (field_name -> [possible_column_names])
        patterns = {
            "Recording Path": ["path", "file_path", "recording_path", "filename", "file"],
            "Cell Name": ["cell", "cell_name", "cell_id", "name"],
            "Protocol": ["protocol", "experiment", "condition", "stim", "stimulus"],
            "Spike Count": ["spike_count", "n_spikes", "spikes", "count"],
            "Spike Rate": ["spike_rate", "firing_rate", "rate", "frequency", "hz"],
            "ISI Mean": ["isi_mean", "mean_isi", "isi_avg", "avg_isi"],
            "ISI CV": ["isi_cv", "cv_isi", "isi_variability"],
            "First Spike Latency": ["first_spike_latency", "latency", "first_spike"],
            "Sweep Number": ["sweep", "sweep_number", "trial", "trial_number"]
        }
        
        # Auto-detect mappings
        for field_name, possible_names in patterns.items():
            combo = self.field_combos.get(field_name)
            if combo:
                for column in all_columns:
                    column_lower = column.lower()
                    for pattern in possible_names:
                        if pattern in column_lower or column_lower in pattern:
                            # Find the index of this column in the combo box
                            for i in range(combo.count()):
                                if combo.itemData(i) == column:
                                    combo.setCurrentIndex(i)
                                    break
                            break
    
    def validatePage(self):
        """Validate that essential mappings are set"""
        wizard = self.wizard()
        
        # Store mappings
        wizard.column_mappings = {}
        for field_name, combo in self.field_combos.items():
            column = combo.currentData()
            if column:
                wizard.column_mappings[field_name] = column
        
        # Check for essential mappings
        if "Recording Path" not in wizard.column_mappings:
            QMessageBox.warning(self, "Missing Mapping", 
                              "Recording Path mapping is required to proceed.")
            return False
        
        return True

class ImportOptionsPage(QWizardPage):
    """Page for configuring import options"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Import Options")
        self.setSubTitle("Configure how the data should be imported")
        
        layout = QVBoxLayout()
        
        # Cell handling options
        cell_group = QGroupBox("Cell Handling")
        cell_layout = QFormLayout()
        
        self.create_cells_check = QCheckBox("Create new cells if they don't exist")
        self.create_cells_check.setChecked(True)
        cell_layout.addRow(self.create_cells_check)
        
        self.update_existing_check = QCheckBox("Update existing cell data")
        self.update_existing_check.setChecked(True)
        cell_layout.addRow(self.update_existing_check)
        
        cell_group.setLayout(cell_layout)
        layout.addWidget(cell_group)
        
        # Protocol handling
        protocol_group = QGroupBox("Protocol Handling")
        protocol_layout = QFormLayout()
        
        self.create_protocols_check = QCheckBox("Create new protocols if they don't exist")
        self.create_protocols_check.setChecked(True)
        protocol_layout.addRow(self.create_protocols_check)
        
        self.default_protocol_edit = QLineEdit("Unknown")
        protocol_layout.addRow("Default protocol name:", self.default_protocol_edit)
        
        protocol_group.setLayout(protocol_layout)
        layout.addWidget(protocol_group)
        
        # Data validation
        validation_group = QGroupBox("Data Validation")
        validation_layout = QFormLayout()
        
        self.skip_invalid_check = QCheckBox("Skip rows with invalid data")
        self.skip_invalid_check.setChecked(True)
        validation_layout.addRow(self.skip_invalid_check)
        
        self.log_errors_check = QCheckBox("Log import errors")
        self.log_errors_check.setChecked(True)
        validation_layout.addRow(self.log_errors_check)
        
        validation_group.setLayout(validation_layout)
        layout.addWidget(validation_group)
        
        # Advanced options
        advanced_group = QGroupBox("Advanced Options")
        advanced_layout = QFormLayout()
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 1000)
        self.batch_size_spin.setValue(100)
        advanced_layout.addRow("Batch size:", self.batch_size_spin)
        
        advanced_group.setLayout(advanced_layout)
        layout.addWidget(advanced_group)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def validatePage(self):
        """Store import options"""
        wizard = self.wizard()
        wizard.import_options = {
            'create_cells': self.create_cells_check.isChecked(),
            'update_existing': self.update_existing_check.isChecked(),
            'create_protocols': self.create_protocols_check.isChecked(),
            'default_protocol': self.default_protocol_edit.text(),
            'skip_invalid': self.skip_invalid_check.isChecked(),
            'log_errors': self.log_errors_check.isChecked(),
            'batch_size': self.batch_size_spin.value()
        }
        return True

class ImportProgressPage(QWizardPage):
    """Page showing import progress"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Importing Data")
        self.setSubTitle("Please wait while the data is being imported...")
        
        layout = QVBoxLayout()
        
        # Progress bars
        self.overall_progress = QProgressBar()
        layout.addWidget(QLabel("Overall Progress:"))
        layout.addWidget(self.overall_progress)
        
        self.file_progress = QProgressBar()
        layout.addWidget(QLabel("Current File:"))
        layout.addWidget(self.file_progress)
        
        # Status label
        self.status_label = QLabel("Preparing import...")
        layout.addWidget(self.status_label)
        
        # Log area
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(200)
        layout.addWidget(QLabel("Import Log:"))
        layout.addWidget(self.log_text)
        
        self.setLayout(layout)
        
        # Track import state
        self.import_completed = False
        self.import_successful = False
    
    def initializePage(self):
        """Start the import process"""
        self.import_completed = False
        self.import_successful = False
        self.log_text.clear()
        
        # Start import in a separate thread (simplified for now)
        self.performImport()
    
    def performImport(self):
        """Perform the actual data import"""
        wizard = self.wizard()
        
        try:
            total_files = len(wizard.csv_files)
            self.overall_progress.setMaximum(total_files)
            
            imported_count = 0
            error_count = 0
            
            for file_idx, file_path in enumerate(wizard.csv_files):
                self.status_label.setText(f"Processing {os.path.basename(file_path)}...")
                self.overall_progress.setValue(file_idx)
                
                try:
                    # Load the full CSV file
                    df = pd.read_csv(file_path)
                    self.file_progress.setMaximum(len(df))
                    
                    self.log_text.append(f"Processing {os.path.basename(file_path)} ({len(df)} rows)")
                    
                    # Process each row
                    for row_idx, row in df.iterrows():
                        self.file_progress.setValue(row_idx + 1);
                        
                        try:
                            # Extract data based on mappings
                            row_data = {}
                            for field_name, column_name in wizard.column_mappings.items():
                                if column_name in df.columns:
                                    row_data[field_name] = row[column_name]
                            
                            # Import the row data
                            success = wizard.database.import_spike_data_row(
                                row_data, wizard.import_options
                            )
                            
                            if success:
                                imported_count += 1
                            else:
                                error_count += 1
                                if wizard.import_options.get('log_errors', True):
                                    self.log_text.append(f"Warning: Row {row_idx + 1} - import failed")
                        
                        except Exception as e:
                            error_count += 1
                            if wizard.import_options.get('log_errors', True):
                                self.log_text.append(f"Error: Row {row_idx + 1} - {str(e)}")
                            
                            if not wizard.import_options.get('skip_invalid', True):
                                raise
                
                except Exception as e:
                    self.log_text.append(f"Error processing file {os.path.basename(file_path)}: {str(e)}")
                    if not wizard.import_options.get('skip_invalid', True):
                        raise
            
            self.overall_progress.setValue(total_files)
            self.status_label.setText("Import completed!")
            
            self.log_text.append(f"\nImport Summary:")
            self.log_text.append(f"Successfully imported: {imported_count} rows")
            self.log_text.append(f"Errors encountered: {error_count} rows")
            
            self.import_successful = True;
            
        except Exception as e:
            self.status_label.setText("Import failed!")
            self.log_text.append(f"\nImport failed with error: {str(e)}")
            self.import_successful = False
        
        finally:
            self.import_completed = True
            # Enable the Next button
            self.wizard().button(QWizard.NextButton).setEnabled(True)
    
    def isComplete(self):
        """Page is complete when import is finished"""
        return self.import_completed

class CompletePage(QWizardPage):
    """Final page showing import results"""
    
    def __init__(self):
        super().__init__()
        self.setTitle("Import Complete")
        self.setSubTitle("The spike data import process has finished")
        
        layout = QVBoxLayout()
        
        self.result_label = QLabel()
        layout.addWidget(self.result_label)
        
        self.details_text = QTextEdit()
        self.details_text.setMaximumHeight(150)
        layout.addWidget(self.details_text)
        
        layout.addStretch()
        self.setLayout(layout)
    
    def initializePage(self):
        """Initialize the completion page"""
        import_page = self.wizard().page(SpikeDataImportWizard.PAGE_IMPORT)
        
        if import_page.import_successful:
            self.result_label.setText("✓ Import completed successfully!")
            self.result_label.setStyleSheet("color: green; font-weight: bold;")
            
            self.details_text.setPlainText(
                "The spike analysis data has been successfully imported into your database. "
                "You can now view and analyze the imported data in the main interface."
            )
        else:
            self.result_label.setText("⚠ Import completed with errors")
            self.result_label.setStyleSheet("color: orange; font-weight: bold;")
            
            self.details_text.setPlainText(
                "The import process encountered some errors. Please review the import log "
                "to see which data could not be imported. You may need to fix the source "
                "data and retry the import."
            )


def run():
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = DatabaseBuilder()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())