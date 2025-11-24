"""
File Browser Widget for CSV/Excel Editor
Provides file system navigation with drag support
"""

from PySide2.QtWidgets import (QWidget, QVBoxLayout, QTreeView, 
                               QFileSystemModel, QHeaderView, QLineEdit,
                               QHBoxLayout, QPushButton, QLabel)
from PySide2.QtCore import QDir, Qt, Signal
from PySide2.QtGui import QIcon
import os


class FileBrowser(QWidget):
    """File browser widget with drag support for files"""
    
    file_selected = Signal(str)  # file_path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.setup_model()
        
    def setup_ui(self):
        """Setup the user interface"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        
        # Path navigation
        nav_layout = QHBoxLayout()
        
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Enter path...")
        self.path_edit.returnPressed.connect(self.navigate_to_path)
        nav_layout.addWidget(self.path_edit)
        
        self.up_button = QPushButton("↑")
        self.up_button.setMaximumWidth(30)
        self.up_button.setToolTip("Go up one directory")
        self.up_button.clicked.connect(self.go_up)
        nav_layout.addWidget(self.up_button)
        
        self.home_button = QPushButton("🏠")
        self.home_button.setMaximumWidth(30)
        self.home_button.setToolTip("Go to home directory")
        self.home_button.clicked.connect(self.go_home)
        nav_layout.addWidget(self.home_button)
        
        layout.addLayout(nav_layout)
        
        # File tree
        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(False)
        self.tree_view.setAlternatingRowColors(True)
        self.tree_view.setDragEnabled(True)
        self.tree_view.setDefaultDropAction(Qt.CopyAction)
        self.tree_view.setSelectionBehavior(QTreeView.SelectRows)
        
        # Connect double-click to navigate
        self.tree_view.doubleClicked.connect(self.on_double_click)
        
        layout.addWidget(self.tree_view)
        
        # Current path label
        self.current_path_label = QLabel()
        self.current_path_label.setWordWrap(True)
        self.current_path_label.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(self.current_path_label)
        
    def setup_model(self):
        """Setup the file system model"""
        self.model = QFileSystemModel()
        self.model.setRootPath(QDir.currentPath())
        
        # Set filters to show common file types
        self.model.setNameFilters([
            "*.csv", "*.xlsx", "*.xls", "*.txt", "*.abf", "*.h5", "*.hdf5"
        ])
        self.model.setNameFilterDisables(False)
        
        self.tree_view.setModel(self.model)
        
        # Set initial root
        self.set_root_path(QDir.currentPath())
        
        # Hide some columns for cleaner look
        self.tree_view.setColumnHidden(1, True)  # Size
        self.tree_view.setColumnHidden(2, True)  # Type
        self.tree_view.setColumnHidden(3, True)  # Date modified
        
        # Resize columns
        header = self.tree_view.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        
    def set_root_path(self, path):
        """Set the root path for the file browser"""
        if os.path.exists(path):
            self.model.setRootPath(path)
            root_index = self.model.index(path)
            self.tree_view.setRootIndex(root_index)
            
            # Update UI
            self.path_edit.setText(path)
            self.current_path_label.setText(f"Current: {path}")
            
            # Enable/disable up button
            parent_path = os.path.dirname(path)
            self.up_button.setEnabled(parent_path != path)
            
    def navigate_to_path(self):
        """Navigate to path entered in line edit"""
        path = self.path_edit.text().strip()
        if path and os.path.exists(path):
            if os.path.isfile(path):
                path = os.path.dirname(path)
            self.set_root_path(path)
        else:
            # Reset to current path if invalid
            current_path = self.model.rootPath()
            self.path_edit.setText(current_path)
            
    def go_up(self):
        """Go up one directory level"""
        current_path = self.model.rootPath()
        parent_path = os.path.dirname(current_path)
        if parent_path != current_path:  # Not at root
            self.set_root_path(parent_path)
            
    def go_home(self):
        """Go to home directory"""
        home_path = QDir.homePath()
        self.set_root_path(home_path)
        
    def on_double_click(self, index):
        """Handle double-click on items"""
        file_path = self.model.filePath(index)
        
        if os.path.isdir(file_path):
            # Navigate to directory
            self.set_root_path(file_path)
        else:
            # Emit signal for file selection
            self.file_selected.emit(file_path)
            
    def get_current_path(self):
        """Get current browsed path"""
        return self.model.rootPath()
        
    def refresh(self):
        """Refresh the current directory"""
        current_path = self.model.rootPath()
        self.model.setRootPath("")  # Clear
        self.model.setRootPath(current_path)  # Reload