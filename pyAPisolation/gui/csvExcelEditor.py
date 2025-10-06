"""
CSV/Excel Editor Application
Main application class for editing spreadsheets with file drag-and-drop support
"""

from . import csvEditorBase
from .dragDropTable import DragDropTableWidget
from .fileBrowser import FileBrowser

from PySide2.QtWidgets import (QApplication, QMainWindow, QFileDialog, 
                               QVBoxLayout, QWidget, QMessageBox, 
                               QTreeView, QFileSystemModel, QHeaderView,
                               QHBoxLayout, QProgressDialog)
from PySide2.QtCore import QDir, Qt, QThread, Signal
from PySide2.QtGui import QFont, QKeySequence
import pandas as pd
import numpy as np
import sys
import os
import traceback


class FileIOThread(QThread):
    """Background thread for file I/O operations"""
    
    finished = Signal(object)  # data
    error = Signal(str)  # error message
    progress = Signal(int)  # progress percentage
    
    def __init__(self, file_path, operation, data=None):
        super().__init__()
        self.file_path = file_path
        self.operation = operation  # 'load' or 'save'
        self.data = data
        
    def run(self):
        try:
            if self.operation == 'load':
                self._load_file()
            elif self.operation == 'save':
                self._save_file()
        except Exception as e:
            self.error.emit(str(e))
            
    def _load_file(self):
        """Load file in background"""
        self.progress.emit(10)
        
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        self.progress.emit(30)
        
        if file_ext == '.csv':
            data = pd.read_csv(self.file_path)
        elif file_ext in ['.xlsx', '.xls']:
            data = pd.read_excel(self.file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        self.progress.emit(80)
        
        # Convert all data to strings to handle mixed types
        data = data.astype(str)
        
        self.progress.emit(100)
        self.finished.emit(data)
        
    def _save_file(self):
        """Save file in background"""
        self.progress.emit(10)
        
        file_ext = os.path.splitext(self.file_path)[1].lower()
        
        self.progress.emit(30)
        
        if file_ext == '.csv':
            self.data.to_csv(self.file_path, index=False)
        elif file_ext in ['.xlsx', '.xls']:
            self.data.to_excel(self.file_path, index=False)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
        self.progress.emit(100)
        self.finished.emit(True)


class CSVExcelEditor(csvEditorBase.Ui_csvEditorBase):
    """Main CSV/Excel editor application"""
    
    def __init__(self):
        super().__init__()
        self.current_file = None
        self.is_modified = False
        self.io_thread = None
        
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow = MainWindow
        
        # Setup custom table widget
        self.setup_table()
        
        # Setup file browser
        self.setup_file_browser()
        
        # Connect menu actions
        self.connect_actions()
        
        # Setup status
        self.update_status("Ready")
        
        # Enable keyboard shortcuts
        self.setup_shortcuts()
        
    def setup_table(self):
        """Setup the custom drag-drop table widget"""
        # Create table widget
        self.table = DragDropTableWidget()
        
        # Add to frame
        layout = QVBoxLayout(self.tableFrame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.table)
        
        # Connect signals
        self.table.file_dropped.connect(self.on_file_dropped)
        self.table.data_changed.connect(self.on_data_changed)
        
        # Connect button signals
        self.addRowButton.clicked.connect(self.table.add_row)
        self.addColumnButton.clicked.connect(self.table.add_column)
        self.deleteRowButton.clicked.connect(self.table.delete_selected_row)
        self.deleteColumnButton.clicked.connect(self.table.delete_selected_column)
        
    def setup_file_browser(self):
        """Setup file browser panel"""
        try:
            # Use the existing FileBrowser if available
            self.file_browser = FileBrowser()
            
            # Add to frame
            layout = QVBoxLayout(self.fileBrowserFrame)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.file_browser)
            
        except ImportError:
            # Fallback to basic file system model
            self.file_model = QFileSystemModel()
            self.file_model.setRootPath(QDir.currentPath())
            
            self.file_tree = QTreeView()
            self.file_tree.setModel(self.file_model)
            self.file_tree.setRootIndex(self.file_model.index(QDir.currentPath()))
            
            # Hide size, type, date columns for simplicity
            self.file_tree.setColumnHidden(1, True)
            self.file_tree.setColumnHidden(2, True)
            self.file_tree.setColumnHidden(3, True)
            
            # Enable drag from file tree
            self.file_tree.setDragEnabled(True)
            self.file_tree.setDefaultDropAction(Qt.CopyAction)
            
            # Add to frame
            layout = QVBoxLayout(self.fileBrowserFrame)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self.file_tree)
            
    def setup_shortcuts(self):
        """Setup keyboard shortcuts"""
        self.actionNew.setShortcut(QKeySequence.New)
        self.actionOpen.setShortcut(QKeySequence.Open)
        self.actionSave.setShortcut(QKeySequence.Save)
        self.actionSaveAs.setShortcut(QKeySequence.SaveAs)
        
    def connect_actions(self):
        """Connect menu actions to methods"""
        self.actionNew.triggered.connect(self.new_file)
        self.actionOpen.triggered.connect(self.open_file)
        self.actionSave.triggered.connect(self.save_file)
        self.actionSaveAs.triggered.connect(self.save_file_as)
        self.actionExit.triggered.connect(self.MainWindow.close)
        
    def on_file_dropped(self, row, col, file_path):
        """Handle file dropped into table cell"""
        self.update_status(f"File dropped: {os.path.basename(file_path)} "
                          f"at row {row+1}, column {col+1}")
        self.is_modified = True
        self.update_window_title()
        
    def on_data_changed(self):
        """Handle table data changes"""
        self.is_modified = True
        self.update_window_title()
        
    def new_file(self):
        """Create new file"""
        if self.check_unsaved_changes():
            self.table.clear_all()
            self.current_file = None
            self.is_modified = False
            self.update_window_title()
            self.update_status("New file created")
            
    def open_file(self):
        """Open file dialog and load file"""
        if not self.check_unsaved_changes():
            return
            
        file_path, _ = QFileDialog.getOpenFileName(
            self.MainWindow,
            "Open CSV/Excel File",
            QDir.currentPath(),
            "Spreadsheet Files (*.csv *.xlsx *.xls);;CSV Files (*.csv);;"
            "Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if file_path:
            self.load_file(file_path)
            
    def load_file(self, file_path):
        """Load file in background thread"""
        # Show progress dialog
        self.progress_dialog = QProgressDialog(
            f"Loading {os.path.basename(file_path)}...", 
            "Cancel", 0, 100, self.MainWindow
        )
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Start background thread
        self.io_thread = FileIOThread(file_path, 'load')
        self.io_thread.progress.connect(self.progress_dialog.setValue)
        self.io_thread.finished.connect(self.on_file_loaded)
        self.io_thread.error.connect(self.on_file_error)
        self.io_thread.start()
        
    def on_file_loaded(self, data):
        """Handle successful file loading"""
        self.progress_dialog.close()
        
        try:
            self.table.from_dataframe(data)
            self.current_file = self.io_thread.file_path
            self.is_modified = False
            self.update_window_title()
            self.update_status(f"Loaded: {os.path.basename(self.current_file)}")
        except Exception as e:
            self.show_error(f"Error loading file: {str(e)}")
            
    def on_file_error(self, error_msg):
        """Handle file loading error"""
        self.progress_dialog.close()
        self.show_error(f"Error loading file: {error_msg}")
        
    def save_file(self):
        """Save current file"""
        if self.current_file:
            self.save_to_file(self.current_file)
        else:
            self.save_file_as()
            
    def save_file_as(self):
        """Save file as dialog"""
        file_path, _ = QFileDialog.getSaveFileName(
            self.MainWindow,
            "Save CSV/Excel File",
            QDir.currentPath(),
            "CSV Files (*.csv);;Excel Files (*.xlsx);;All Files (*)"
        )
        
        if file_path:
            self.save_to_file(file_path)
            
    def save_to_file(self, file_path):
        """Save file in background thread"""
        # Get data from table
        data = self.table.to_dataframe()
        
        # Show progress dialog
        self.progress_dialog = QProgressDialog(
            f"Saving {os.path.basename(file_path)}...", 
            "Cancel", 0, 100, self.MainWindow
        )
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.show()
        
        # Start background thread
        self.io_thread = FileIOThread(file_path, 'save', data)
        self.io_thread.progress.connect(self.progress_dialog.setValue)
        self.io_thread.finished.connect(self.on_file_saved)
        self.io_thread.error.connect(self.on_file_error)
        self.io_thread.start()
        
    def on_file_saved(self, success):
        """Handle successful file saving"""
        self.progress_dialog.close()
        
        if success:
            self.current_file = self.io_thread.file_path
            self.is_modified = False
            self.update_window_title()
            self.update_status(f"Saved: {os.path.basename(self.current_file)}")
        else:
            self.show_error("Error saving file")
            
    def check_unsaved_changes(self):
        """Check for unsaved changes and prompt user"""
        if self.is_modified:
            reply = QMessageBox.question(
                self.MainWindow,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before continuing?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            
            if reply == QMessageBox.Save:
                self.save_file()
                return not self.is_modified  # Return False if save was cancelled
            elif reply == QMessageBox.Cancel:
                return False
                
        return True
        
    def update_window_title(self):
        """Update window title with current file and modification status"""
        title = "CSV/Excel Editor - pyAPisolation"
        
        if self.current_file:
            filename = os.path.basename(self.current_file)
            title = f"{filename} - {title}"
            
        if self.is_modified:
            title = f"*{title}"
            
        self.MainWindow.setWindowTitle(title)
        
    def update_status(self, message):
        """Update status bar and status text"""
        #self.MainWindow.statusbar.showMessage(message)
        self.statusText.append(f"[{pd.Timestamp.now().strftime('%H:%M:%S')}] {message}")
        
        # Auto-scroll to bottom
        cursor = self.statusText.textCursor()
        cursor.movePosition(cursor.End)
        self.statusText.setTextCursor(cursor)
        
    def show_error(self, message):
        """Show error message"""
        QMessageBox.critical(self.MainWindow, "Error", message)
        self.update_status(f"Error: {message}")
        
    def closeEvent(self, event):
        """Handle window close event"""
        if self.check_unsaved_changes():
            event.accept()
        else:
            event.ignore()


def main():
    """Main function to run the CSV/Excel editor"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("CSV/Excel Editor")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("pyAPisolation")
    
    # Create main window
    MainWindow = QMainWindow()
    ui = CSVExcelEditor()
    ui.setupUi(MainWindow)
    
    # Override close event
    MainWindow.closeEvent = ui.closeEvent
    
    # Show window
    MainWindow.show()
    
    # Start event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()