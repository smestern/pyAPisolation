from . import prism_writer
print("Loaded basic libraries; importing QT")
from PySide2.QtWidgets import (QApplication, QWidget, QFileDialog, QVBoxLayout,
    QHBoxLayout, QPushButton, QListWidget, QAbstractItemView, QLabel, 
    QLineEdit, QGroupBox, QTextEdit, QTableWidget, QTableWidgetItem, 
    QSplitter, QFrame)
from PySide2.QtCore import Qt
import pandas as pd
import copy
import os

class PrismWriterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.df = pd.DataFrame()
        self.prism_writer = None
        self.file_path = None
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Prism Writer - Advanced Table Creator")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel for controls
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Right panel for preview and results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Add panels to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 800])  # Set initial sizes
        
        # Create all sections
        self.create_file_section(left_layout)
        self.create_data_section(left_layout)
        self.create_grouping_section(left_layout)
        self.create_table_creation_section(left_layout)
        
        self.create_preview_section(right_layout)
        self.create_results_section(right_layout)
        
        # Add stretch to left panel
        left_layout.addStretch()
        
        self.show()
    
    def create_file_section(self, parent_layout):
        """Create file management section"""
        file_group = QGroupBox("File Management")
        file_layout = QVBoxLayout()
        
        # File path display
        self.file_path_label = QLabel("No Prism file selected")
        self.file_path_label.setStyleSheet("color: gray; font-style: italic;")
        file_layout.addWidget(self.file_path_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.new_file_button = QPushButton("New Prism File")
        self.new_file_button.clicked.connect(self.new_file)
        button_layout.addWidget(self.new_file_button)
        
        self.save_file_button = QPushButton("Save File")
        self.save_file_button.clicked.connect(self.save_file)
        self.save_file_button.setEnabled(False)
        button_layout.addWidget(self.save_file_button)
        
        file_layout.addLayout(button_layout)
        file_group.setLayout(file_layout)
        parent_layout.addWidget(file_group)
    
    def create_data_section(self, parent_layout):
        """Create data loading section"""
        data_group = QGroupBox("Data Source")
        data_layout = QVBoxLayout()
        
        # Data file info
        self.data_file_label = QLabel("No data file loaded")
        self.data_file_label.setStyleSheet("color: gray; font-style: italic;")
        data_layout.addWidget(self.data_file_label)
        
        # Open data button
        self.open_csv_button = QPushButton("Load Data (CSV/Excel)")
        self.open_csv_button.clicked.connect(self.open_csv)
        data_layout.addWidget(self.open_csv_button)
        
        # Data info labels
        self.data_info_label = QLabel("")
        data_layout.addWidget(self.data_info_label)
        
        data_group.setLayout(data_layout)
        parent_layout.addWidget(data_group)
    
    def create_grouping_section(self, parent_layout):
        """Create grouping controls section"""
        grouping_group = QGroupBox("Table Grouping Configuration")
        grouping_layout = QVBoxLayout()
        
        # Main group (single selection)
        main_group_frame = QFrame()
        main_layout = QVBoxLayout(main_group_frame)
        main_layout.addWidget(QLabel("Main Group Column (Y-axis groups):"))
        self.main_group_list = QListWidget()
        self.main_group_list.setMaximumHeight(80)
        self.main_group_list.itemSelectionChanged.connect(self.update_preview)
        main_layout.addWidget(self.main_group_list)
        
        self.main_group_info = QLabel("")
        self.main_group_info.setStyleSheet("font-size: 10px; color: #666;")
        main_layout.addWidget(self.main_group_info)
        
        # Sub group (multiple selection)
        sub_group_frame = QFrame()
        sub_layout = QVBoxLayout(sub_group_frame)
        sub_layout.addWidget(QLabel("Sub Group (Sub-columns):"))
        self.sub_group_list = QListWidget()
        self.sub_group_list.setMaximumHeight(80)
        self.sub_group_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.sub_group_list.itemSelectionChanged.connect(self.update_preview)
        sub_layout.addWidget(self.sub_group_list)
        
        self.sub_group_info = QLabel("")
        self.sub_group_info.setStyleSheet("font-size: 10px; color: #666;")
        sub_layout.addWidget(self.sub_group_info)
        
        # Row group (multiple selection)
        row_group_frame = QFrame()
        row_layout = QVBoxLayout(row_group_frame)
        row_layout.addWidget(QLabel("Row Group (Row labels):"))
        self.row_group_list = QListWidget()
        self.row_group_list.setMaximumHeight(80)
        self.row_group_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.row_group_list.itemSelectionChanged.connect(self.update_preview)
        row_layout.addWidget(self.row_group_list)
        
        self.row_group_info = QLabel("")
        self.row_group_info.setStyleSheet("font-size: 10px; color: #666;")
        row_layout.addWidget(self.row_group_info)
        
        # Data columns (multiple selection)
        data_col_frame = QFrame()
        data_layout = QVBoxLayout(data_col_frame)
        data_layout.addWidget(QLabel("Data Columns:"))
        self.data_col_list = QListWidget()
        self.data_col_list.setMaximumHeight(80)
        self.data_col_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.data_col_list.itemSelectionChanged.connect(self.update_preview)
        data_layout.addWidget(self.data_col_list)
        
        self.data_col_info = QLabel("")
        self.data_col_info.setStyleSheet("font-size: 10px; color: #666;")
        data_layout.addWidget(self.data_col_info)
        
        grouping_layout.addWidget(main_group_frame)
        grouping_layout.addWidget(sub_group_frame)
        grouping_layout.addWidget(row_group_frame)
        grouping_layout.addWidget(data_col_frame)
        
        grouping_group.setLayout(grouping_layout)
        parent_layout.addWidget(grouping_group)
    
    def create_table_creation_section(self, parent_layout):
        """Create table creation section"""
        creation_group = QGroupBox("Table Creation")
        creation_layout = QVBoxLayout()
        
        # Table name
        creation_layout.addWidget(QLabel("Table Name:"))
        self.group_table_name = QLineEdit("Group Table Name")
        self.group_table_name.textChanged.connect(self.update_preview)
        creation_layout.addWidget(self.group_table_name)
        
        # Create button
        self.create_group_table_button = QPushButton("Create Group Table")
        self.create_group_table_button.clicked.connect(self.create_group_table)
        self.create_group_table_button.setEnabled(False)
        creation_layout.addWidget(self.create_group_table_button)
        
        # Validation info
        self.validation_label = QLabel("")
        self.validation_label.setWordWrap(True)
        creation_layout.addWidget(self.validation_label)
        
        creation_group.setLayout(creation_layout)
        parent_layout.addWidget(creation_group)
    
    def create_preview_section(self, parent_layout):
        """Create data preview section"""
        preview_group = QGroupBox("Data Preview & Table Structure")
        preview_layout = QVBoxLayout()
        
        # Data preview table
        preview_layout.addWidget(QLabel("Data Preview (first 10 rows):"))
        self.preview_table = QTableWidget()
        self.preview_table.setMaximumHeight(250)
        preview_layout.addWidget(self.preview_table)
        
        # Table structure preview
        preview_layout.addWidget(QLabel("Prism Table Structure Preview:"))
        self.structure_preview = QTextEdit()
        self.structure_preview.setMaximumHeight(200)
        self.structure_preview.setReadOnly(True)
        self.structure_preview.setStyleSheet("font-family: monospace; background-color: #f5f5f5;")
        preview_layout.addWidget(self.structure_preview)
        
        preview_group.setLayout(preview_layout)
        parent_layout.addWidget(preview_group)
    
    def create_results_section(self, parent_layout):
        """Create results section"""
        results_group = QGroupBox("Created Tables")
        results_layout = QVBoxLayout()
        
        # Tables list
        self.table_list = QListWidget()
        results_layout.addWidget(self.table_list)
        
        # Delete button
        self.delete_table_button = QPushButton("Delete Selected Table")
        self.delete_table_button.clicked.connect(self.delete_table)
        results_layout.addWidget(self.delete_table_button)
        
        results_group.setLayout(results_layout)
        parent_layout.addWidget(results_group)

    def new_file(self):
        """Create a new Prism file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self, 
            "Create New Prism File", 
            "", 
            "Prism Files (*.pzfx)"
        )
        if file_path:
            self.file_path = file_path
            self.prism_writer = prism_writer.PrismFile()
            self.save_file_button.setEnabled(True)
            
            # Update UI
            filename = os.path.basename(file_path)
            self.file_path_label.setText(f"File: {filename}")
            self.file_path_label.setStyleSheet("color: green; font-weight: bold;")
            self.table_list.clear()
            self.update_validation()

    def save_file(self):
        """Save the current Prism file"""
        if self.prism_writer and self.file_path:
            try:
                self.prism_writer.save(self.file_path)
                self.file_path_label.setText(f"✓ Saved: {os.path.basename(self.file_path)}")
                self.file_path_label.setStyleSheet("color: green; font-weight: bold;")
            except Exception as e:
                self.file_path_label.setText(f"Error saving: {str(e)}")
                self.file_path_label.setStyleSheet("color: red; font-weight: bold;")

    def open_csv(self):
        """Load data from CSV or Excel file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Load Data File", 
            "", 
            "Data files (*.csv *.xlsx *.xls);;CSV files (*.csv);;Excel files (*.xlsx *.xls)"
        )
        if file_path:
            try:
                # Load data
                if file_path.endswith('.csv'):
                    self.df = pd.read_csv(file_path)
                else:
                    self.df = pd.read_excel(file_path)
                
                # Update UI
                filename = os.path.basename(file_path)
                self.data_file_label.setText(f"Loaded: {filename}")
                self.data_file_label.setStyleSheet("color: green; font-weight: bold;")
                
                # Update data info
                rows, cols = self.df.shape
                self.data_info_label.setText(f"Shape: {rows} rows × {cols} columns")
                
                # Update column lists
                self.update_column_lists()
                
                # Update preview
                self.update_data_preview()
                
                # Update validation
                self.update_validation()
                
            except Exception as e:
                self.data_file_label.setText(f"Error loading file: {str(e)}")
                self.data_file_label.setStyleSheet("color: red; font-weight: bold;")
                self.df = None
    
    def update_column_lists(self):
        """Update all column selection lists"""
        if self.df is None:
            return
        
        columns = list(self.df.columns)
        
        # Clear all lists
        for list_widget in [self.main_group_list, self.sub_group_list, 
                           self.row_group_list, self.data_col_list]:
            list_widget.clear()
        
        # Analyze column types for better recommendations
        numeric_cols = []
        categorical_cols = []
        
        for col in columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Populate lists with type indicators
        for col in columns:
            col_type = "[NUM]" if col in numeric_cols else "[CAT]"
            display_text = f"{col_type} {col}"
            
            for list_widget in [self.main_group_list, self.sub_group_list, 
                               self.row_group_list, self.data_col_list]:
                list_widget.addItem(display_text)
    
    def update_data_preview(self, df=None):
        """Update the data preview table"""
        if df is not None:
            df = df.copy()
        elif df is None and self.df is not None:
            df = self.df.copy()
        elif self.df is None:
            self.preview_table.clear()
            return
        
        # Show first 10 rows and limit columns if too many
        preview_data = df.head(10)
        max_cols = min(8, len(preview_data.columns))
        display_data = preview_data.iloc[:, :max_cols]
        
        self.preview_table.setRowCount(len(display_data))
        self.preview_table.setColumnCount(len(display_data.columns))
        
        # Set headers
        self.preview_table.setHorizontalHeaderLabels(
            [str(col) for col in display_data.columns]
        )
        
        # Fill data
        for i, (_, row) in enumerate(display_data.iterrows()):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.preview_table.setItem(i, j, item)
        
        self.preview_table.resizeColumnsToContents()
    
    def update_preview(self):
        """Update the table structure preview"""
        if self.df is None:
            self.structure_preview.setText("No data loaded")
            return
        
        # Get selected items
        main_group = self.get_selected_columns(self.main_group_list)
        sub_group = self.get_selected_columns(self.sub_group_list)
        row_group = self.get_selected_columns(self.row_group_list)
        data_cols = self.get_selected_columns(self.data_col_list)
        
        # Update info labels with category counts
        self.update_group_info_labels(main_group, sub_group, row_group, data_cols)
        
        # Generate structure preview
        preview_text = self.generate_structure_preview(main_group, sub_group, row_group, data_cols)
        self.structure_preview.setText(preview_text)
        
        # Update validation
        self.update_validation()
    
    def get_selected_columns(self, list_widget):
        """Get selected column names from a list widget"""
        selected_items = list_widget.selectedItems()
        return [item.text().split(' ', 1)[1] for item in selected_items]  # Remove [NUM]/[CAT] prefix
    
    def update_group_info_labels(self, main_group, sub_group, row_group, data_cols):
        """Update info labels showing category counts"""
        def get_unique_count_info(columns):
            if not columns or self.df is None:
                return "No selection"
            
            info_parts = []
            for col in columns:
                if col in self.df.columns:
                    unique_count = self.df[col].nunique()
                    info_parts.append(f"{col}: {unique_count} categories")
            
            return "; ".join(info_parts) if info_parts else "Invalid selection"
        
        self.main_group_info.setText(get_unique_count_info(main_group))
        self.sub_group_info.setText(get_unique_count_info(sub_group))
        self.row_group_info.setText(get_unique_count_info(row_group))
        self.data_col_info.setText(f"{len(data_cols)} columns selected" if data_cols else "No columns selected")
    
    def generate_structure_preview(self, main_group, sub_group, row_group, data_cols):
        """Generate a text preview of the table structure"""
        if not main_group:
            return "Select a main group column to see table structure preview"
        
        preview = f"Table: {self.group_table_name.text()}\n"
        preview += "=" * 50 + "\n\n"
        
        if self.df is None:
            return preview + "No data loaded"
        
        # Main group info
        main_col = main_group[0]
        if main_col in self.df.columns:
            main_groups = self.df[main_col].unique()
            preview += f"Main Groups ({main_col}): {len(main_groups)} groups\n"
            preview += f"  Groups: {', '.join(map(str, main_groups[:5]))}"
            if len(main_groups) > 5:
                preview += f" ... and {len(main_groups) - 5} more"
            preview += "\n\n"
        
        # Sub group info
        if sub_group:
            if len(sub_group) == 1 and sub_group[0] in self.df.columns:
                sub_groups = self.df[sub_group[0]].unique()
                preview += f"Sub Groups ({sub_group[0]}): {len(sub_groups)} sub-columns\n"
                preview += f"  Sub-columns: {', '.join(map(str, sub_groups[:3]))}"
                if len(sub_groups) > 3:
                    preview += f" ... and {len(sub_groups) - 3} more"
                preview += "\n\n"
            else:
                preview += f"Sub Group Columns: {len(sub_group)} columns as sub-groups\n"
                preview += f"  Columns: {', '.join(sub_group[:3])}"
                if len(sub_group) > 3:
                    preview += f" ... and {len(sub_group) - 3} more"
                preview += "\n\n"
        
        # Row group info
        if row_group:
            if len(row_group) == 1 and row_group[0] in self.df.columns:
                row_groups = self.df[row_group[0]].unique()
                preview += f"Row Groups ({row_group[0]}): {len(row_groups)} row labels\n"
                preview += f"  Rows: {', '.join(map(str, row_groups[:3]))}"
                if len(row_groups) > 3:
                    preview += f" ... and {len(row_groups) - 3} more"
                preview += "\n\n"
            else:
                preview += f"Row Group Columns: {len(row_group)} columns as row groups\n"
                preview += f"  Columns: {', '.join(row_group[:3])}"
                if len(row_group) > 3:
                    preview += f" ... and {len(row_group) - 3} more"
                preview += "\n\n"
        
        # Data columns info
        if data_cols:
            preview += f"Data Columns: {len(data_cols)} columns\n"
            preview += f"  Columns: {', '.join(data_cols[:3])}"
            if len(data_cols) > 3:
                preview += f" ... and {len(data_cols) - 3} more"
            preview += "\n\n"
        
        # Estimated table dimensions
        try:
            if main_col in self.df.columns:
                num_main_groups = self.df[main_col].nunique()
                num_sub_cols = len(sub_group) if len(sub_group) > 1 else (
                    self.df[sub_group[0]].nunique() if sub_group and sub_group[0] in self.df.columns else 1
                )
                num_rows = len(row_group) if len(row_group) > 1 else (
                    self.df[row_group[0]].nunique() if row_group and row_group[0] in self.df.columns else 
                    len(self.df)
                )
                
                preview += f"Estimated Table Structure:\n"
                preview += f"  Main groups: {num_main_groups}\n"
                preview += f"  Sub-columns per group: {num_sub_cols}\n"
                preview += f"  Rows: {num_rows}\n"
                preview += f"  Total columns: {num_main_groups * num_sub_cols}\n"
        except:
            preview += "Unable to estimate table dimensions\n"
        
        #generate a table preview using the prism_writer
        #try:
        self.prism_writer.make_group_table(
            group_name="__" + self.group_table_name.text().strip(),
            group_values=self.df.sample(frac=0.1).head(), # sample for preview, since full df may be large
            groupby=main_group,
            cols=data_cols if data_cols else None,
            subgroupcols=sub_group if sub_group else None,
            rowgroupcols=row_group if row_group else None
        )
        #preview += f"Table Preview:\n{preview_table}\n"
        df_preview = self.prism_writer.to_dataframe("__" + self.group_table_name.text().strip())
        #except Exception as e:
        #preview += f"Error generating table preview: {e}\n"
        #replace the self.preview_table with a dataframe preview
        if df_preview is not None:
            self.update_data_preview(df_preview)
        return preview
    
    def update_validation(self):
        """Update validation status and enable/disable create button"""
        if self.df.empty or not self.prism_writer:
            self.validation_label.setText("❌ Prism file and data required")
            self.validation_label.setStyleSheet("color: red;")
            self.create_group_table_button.setEnabled(False)
            return
        
        main_group = self.get_selected_columns(self.main_group_list)
        
        if not main_group:
            self.validation_label.setText("❌ Main group column required")
            self.validation_label.setStyleSheet("color: red;")
            self.create_group_table_button.setEnabled(False)
            return
        
        if len(main_group) > 1:
            self.validation_label.setText("❌ Select only one main group column")
            self.validation_label.setStyleSheet("color: red;")
            self.create_group_table_button.setEnabled(False)
            return
        
        if not self.group_table_name.text().strip():
            self.validation_label.setText("❌ Table name required")
            self.validation_label.setStyleSheet("color: red;")
            self.create_group_table_button.setEnabled(False)
            return
        
        # Check for valid columns
        all_selected = main_group + self.get_selected_columns(self.sub_group_list) + \
                      self.get_selected_columns(self.row_group_list) + \
                      self.get_selected_columns(self.data_col_list)
        
        invalid_cols = [col for col in all_selected if col not in self.df.columns]
        if invalid_cols:
            self.validation_label.setText(f"❌ Invalid columns: {', '.join(invalid_cols)}")
            self.validation_label.setStyleSheet("color: red;")
            self.create_group_table_button.setEnabled(False)
            return
        
        self.validation_label.setText("✅ Ready to create table")
        self.validation_label.setStyleSheet("color: green;")
        self.create_group_table_button.setEnabled(True)


    def create_group_table(self):
        """Create the group table with current settings"""
        if not self.validate_inputs():
            return
        
        try:
            # Get selections
            main_group = self.get_selected_columns(self.main_group_list)[0]
            sub_group = self.get_selected_columns(self.sub_group_list)
            row_group = self.get_selected_columns(self.row_group_list)
            col_group = self.get_selected_columns(self.data_col_list)
            
            # Process selections according to the original logic
            if len(sub_group) > 1:
                sub_group_cols = copy.copy(sub_group)
                sub_group_by = None
            elif len(sub_group) == 1:
                sub_group_cols = None
                sub_group_by = sub_group[0]
            else:
                sub_group_cols = None
                sub_group_by = None

            if len(row_group) > 1:
                row_group_cols = copy.copy(row_group)
                row_group_by = None
            elif len(row_group) == 1:
                row_group_cols = None
                row_group_by = row_group[0]
            else:
                row_group_cols = None
                row_group_by = None
            
            if len(col_group) > 1:
                cols = col_group
            elif len(col_group) == 1:
                cols = col_group[0]
            else:
                cols = None

            # Create the table
            table_name = self.group_table_name.text().strip()
            self.prism_writer.make_group_table(
                table_name, 
                self.df, 
                main_group, 
                cols=cols, 
                subgroupcols=sub_group_cols, 
                rowgroupcols=row_group_cols, 
                subgroupby=sub_group_by, 
                rowgroupby=row_group_by
            )

            # Update UI
            self.table_list.addItem(f"✓ {table_name}")
            self.validation_label.setText(f"✅ Table '{table_name}' created successfully!")
            self.validation_label.setStyleSheet("color: green; font-weight: bold;")
            
            # Clear selections for next table
            self.group_table_name.setText("Group Table Name")
            
        except Exception as e:
            self.validation_label.setText(f"❌ Error creating table: {str(e)}")
            self.validation_label.setStyleSheet("color: red;")
    
    def validate_inputs(self):
        """Validate all inputs before creating table"""
        if self.df.empty:
            self.validation_label.setText("❌ No data loaded")
            self.validation_label.setStyleSheet("color: red;")
            return False
        
        if not self.prism_writer:
            self.validation_label.setText("❌ No Prism file created")
            self.validation_label.setStyleSheet("color: red;")
            return False
        
        main_group = self.get_selected_columns(self.main_group_list)
        if not main_group:
            self.validation_label.setText("❌ Main group column required")
            self.validation_label.setStyleSheet("color: red;")
            return False
        
        if len(main_group) > 1:
            self.validation_label.setText("❌ Select only one main group column")
            self.validation_label.setStyleSheet("color: red;")
            return False
        
        table_name = self.group_table_name.text().strip()
        if not table_name:
            self.validation_label.setText("❌ Table name required")
            self.validation_label.setStyleSheet("color: red;")
            return False
        
        return True

    def delete_table(self):
        """Delete the selected table"""
        selected_items = self.table_list.selectedItems()
        if not selected_items:
            return
        
        try:
            item_text = selected_items[0].text()
            # Remove the checkmark prefix if present
            table_name = item_text.replace("✓ ", "")
            
            self.prism_writer.delete_table(table_name)
            self.table_list.takeItem(self.table_list.currentRow())
            
            self.validation_label.setText(f"✅ Table '{table_name}' deleted")
            self.validation_label.setStyleSheet("color: green;")
            
        except Exception as e:
            self.validation_label.setText(f"❌ Error deleting table: {str(e)}")
            self.validation_label.setStyleSheet("color: red;")


def main():
    """Main function to run the Prism Writer GUI"""
    app = QApplication([])
    
    # Set application properties
    app.setApplicationName("Prism Writer - Advanced Table Creator")
    app.setApplicationVersion("2.0")
    
    # Create and show GUI
    gui = PrismWriterGUI()
    
    app.exec_()


if __name__ == '__main__':
    main()



        