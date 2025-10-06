"""
Custom table widget with drag-and-drop file support for CSV/Excel editor
"""

from PySide2.QtWidgets import (QTableWidget, QTableWidgetItem, QHeaderView, 
                               QAbstractItemView, QApplication, QMenu)
from PySide2.QtCore import Qt, Signal, QMimeData, QUrl
from PySide2.QtGui import QDragEnterEvent, QDropEvent, QContextMenuEvent
import os
import pandas as pd


class DragDropTableWidget(QTableWidget):
    """Custom table widget that supports file drag-and-drop operations"""
    
    file_dropped = Signal(int, int, str)  # row, column, file_path
    data_changed = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.setAlternatingRowColors(True)
        
        # Allow editing
        self.setEditTriggers(QAbstractItemView.DoubleClicked | 
                           QAbstractItemView.EditKeyPressed |
                           QAbstractItemView.AnyKeyPressed)
        
        # Setup headers
        self.horizontalHeader().setStretchLastSection(True)
        self.verticalHeader().setVisible(True)
        
        # Connect item change signal
        self.itemChanged.connect(self._on_item_changed)
        
        # Initialize with default size
        self.setRowCount(10)
        self.setColumnCount(5)
        self._setup_default_headers()
        
    def _setup_default_headers(self):
        """Setup default column headers"""
        headers = [f"Column {i+1}" for i in range(self.columnCount())]
        self.setHorizontalHeaderLabels(headers)
        
    def _on_item_changed(self, item):
        """Handle item changes"""
        self.data_changed.emit()
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        """Handle drag enter events"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)
            
    def dragMoveEvent(self, event):
        """Handle drag move events"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)
            
    def dropEvent(self, event: QDropEvent):
        """Handle drop events - populate cell with file path"""
        if event.mimeData().hasUrls():
            position = event.pos()
            item = self.itemAt(position)
            
            if item is None:
                # Create item if it doesn't exist
                row = self.rowAt(position.y())
                col = self.columnAt(position.x())
                if row >= 0 and col >= 0:
                    item = QTableWidgetItem()
                    self.setItem(row, col, item)
            
            if item is not None:
                row = item.row()
                col = item.column()
                
                # Get the first file from the drop
                urls = event.mimeData().urls()
                if urls:
                    file_path = urls[0].toLocalFile()
                    if os.path.exists(file_path):
                        # Option 1: Store just filename
                        filename = os.path.basename(file_path)
                        item.setText(filename)
                        
                        # Store full path in item data for later use
                        item.setData(Qt.UserRole, file_path)
                        
                        # Add tooltip with full path
                        item.setToolTip(f"Full path: {file_path}")
                        
                        # Emit signal for external handling
                        self.file_dropped.emit(row, col, file_path)
                        
                        event.acceptProposedAction()
                        return
                        
        super().dropEvent(event)
        
    def contextMenuEvent(self, event: QContextMenuEvent):
        """Handle right-click context menu"""
        item = self.itemAt(event.pos())
        if item is not None:
            menu = QMenu(self)
            
            # Add context menu actions
            clear_action = menu.addAction("Clear Cell")
            copy_path_action = menu.addAction("Copy File Path")
            open_location_action = menu.addAction("Open File Location")
            
            # Get file path from item data
            file_path = item.data(Qt.UserRole)
            
            # Disable actions if no file path
            if not file_path or not os.path.exists(file_path):
                copy_path_action.setEnabled(False)
                open_location_action.setEnabled(False)
                
            action = menu.exec_(self.mapToGlobal(event.pos()))
            
            if action == clear_action:
                item.setText("")
                item.setData(Qt.UserRole, None)
                item.setToolTip("")
            elif action == copy_path_action and file_path:
                clipboard = QApplication.clipboard()
                clipboard.setText(file_path)
            elif action == open_location_action and file_path:
                # Open file location in system file manager
                import subprocess
                import platform
                
                folder_path = os.path.dirname(file_path)
                if platform.system() == "Windows":
                    subprocess.run(["explorer", folder_path])
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", folder_path])
                else:  # Linux
                    subprocess.run(["xdg-open", folder_path])
                    
    def add_row(self):
        """Add a new row to the table"""
        self.insertRow(self.rowCount())
        
    def add_column(self):
        """Add a new column to the table"""
        self.insertColumn(self.columnCount())
        # Update header
        new_col = self.columnCount() - 1
        self.setHorizontalHeaderItem(new_col, 
                                   QTableWidgetItem(f"Column {new_col + 1}"))
        
    def delete_selected_row(self):
        """Delete currently selected row(s)"""
        selected_rows = set()
        for item in self.selectedItems():
            selected_rows.add(item.row())
            
        # Remove rows in reverse order to maintain indices
        for row in sorted(selected_rows, reverse=True):
            self.removeRow(row)
            
    def delete_selected_column(self):
        """Delete currently selected column(s)"""
        selected_cols = set()
        for item in self.selectedItems():
            selected_cols.add(item.column())
            
        # Remove columns in reverse order to maintain indices
        for col in sorted(selected_cols, reverse=True):
            self.removeColumn(col)
            
    def get_file_path(self, row, col):
        """Get file path stored in specific cell"""
        item = self.item(row, col)
        if item:
            return item.data(Qt.UserRole)
        return None
        
    def set_file_path(self, row, col, file_path):
        """Set file path for specific cell"""
        item = self.item(row, col)
        if item is None:
            item = QTableWidgetItem()
            self.setItem(row, col, item)
            
        if file_path and os.path.exists(file_path):
            filename = os.path.basename(file_path)
            item.setText(filename)
            item.setData(Qt.UserRole, file_path)
            item.setToolTip(f"Full path: {file_path}")
        else:
            item.setText("")
            item.setData(Qt.UserRole, None)
            item.setToolTip("")
            
    def to_dataframe(self):
        """Convert table contents to pandas DataFrame"""
        data = []
        
        for row in range(self.rowCount()):
            row_data = []
            for col in range(self.columnCount()):
                item = self.item(row, col)
                if item:
                    # Use file path if available, otherwise use display text
                    file_path = item.data(Qt.UserRole)
                    if file_path:
                        row_data.append(file_path)
                    else:
                        row_data.append(item.text())
                else:
                    row_data.append("")
            data.append(row_data)
            
        # Get column headers
        headers = []
        for col in range(self.columnCount()):
            header_item = self.horizontalHeaderItem(col)
            if header_item:
                headers.append(header_item.text())
            else:
                headers.append(f"Column {col + 1}")
                
        return pd.DataFrame(data, columns=headers)
        
    def from_dataframe(self, df):
        """Load data from pandas DataFrame"""
        # Clear existing data
        self.clear()
        
        # Set size
        self.setRowCount(len(df))
        self.setColumnCount(len(df.columns))
        
        # Set headers
        self.setHorizontalHeaderLabels(list(df.columns))
        
        # Populate data
        for row in range(len(df)):
            for col in range(len(df.columns)):
                value = df.iloc[row, col]
                if pd.isna(value):
                    value = ""
                else:
                    value = str(value)
                    
                item = QTableWidgetItem(value)
                
                # Check if value looks like a file path
                if value and os.path.exists(value):
                    filename = os.path.basename(value)
                    item.setText(filename)
                    item.setData(Qt.UserRole, value)
                    item.setToolTip(f"Full path: {value}")
                    
                self.setItem(row, col, item)
                
    def clear_all(self):
        """Clear all table contents"""
        self.clear()
        self.setRowCount(10)
        self.setColumnCount(5)
        self._setup_default_headers()