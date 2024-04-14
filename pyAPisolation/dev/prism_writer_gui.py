import prism_writer
print("Loaded basic libraries; importing QT")
from PySide2.QtWidgets import QApplication, QWidget, QFileDialog, QVBoxLayout,\
QHBoxLayout, QProgressDialog, QMainWindow, QAction, QTableView, QPushButton, QListWidget, QAbstractItemView, QLabel, QLineEdit
from PySide2.QtCore import QFile, QAbstractTableModel, Qt, QModelIndex
from PySide2 import QtGui
import PySide2.QtCore as QtCore
import pandas as pd
import copy

class PrismWriterGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle("Prism Writer")
        self.setGeometry(100, 100, 800, 600)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.createmain()
        self.show()
    
    def createmain(self):
        self.main_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.layout.addLayout(self.main_layout)
        self.buttons_layout = QHBoxLayout()
        #make the buttons, first a new file button
        self.new_file_button = QPushButton("New File", self)
        self.new_file_button.clicked.connect(self.new_file)
        #make the save file button, disabled until a file is created
        self.save_file_button = QPushButton("Save File", self)
        #disable the save file button until a file is created
        self.save_file_button.setDisabled(True)
        #add the buttons to the layout
        self.buttons_layout.addWidget(self.new_file_button)
        #self.main_layout.addWidget(self.open_file_button)
        self.buttons_layout.addWidget(self.save_file_button)
        #add the layout to the main layout
        self.left_layout.addLayout(self.buttons_layout)

        #create a button to open a csv
        self.open_csv_button = QPushButton("Open CSV", self)
        self.open_csv_button.clicked.connect(self.open_csv)
        self.left_layout.addWidget(self.open_csv_button)

        #create three list widgets, one to group by the main columns, one to group by the sub columns, and one to group by the rows
        self.main_group_list = QListWidget()
        self.sub_group_list = QListWidget()
        self.row_group_list =QListWidget()
        #make them multi select
        #self.main_group_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.sub_group_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.row_group_list.setSelectionMode(QAbstractItemView.MultiSelection)
        #add them to the main layout, with a label and spacer
        self.left_layout.addWidget(QLabel("Main Group"))
        self.left_layout.addWidget(self.main_group_list)
        self.left_layout.addWidget(QLabel("Sub Group"))
        self.left_layout.addWidget(self.sub_group_list)
        self.left_layout.addWidget(QLabel("Row Group"))
        self.left_layout.addWidget(self.row_group_list)

        #creat a text box to name the group table
        self.group_table_name = QLineEdit("Group Table Name")
        self.left_layout.addWidget(self.group_table_name)

        #finally add a button to create the group table
        self.create_group_table_button = QPushButton("Create Group Table", self)
        self.create_group_table_button.clicked.connect(self.create_group_table)
        self.left_layout.addWidget(self.create_group_table_button)
        self.main_layout.addLayout(self.left_layout)
        #make a right layout for the table view
        self.right_layout = QVBoxLayout()
        #make a label that lists the currently open files
        self.prism_title = QLabel("Prism Writer")
        self.right_layout.addWidget(self.prism_title)
        self.table_list_label = QLabel("Tables")
        self.right_layout.addWidget(self.table_list_label)
        #make a list widget that lists the currentlt generated tables
        self.table_list = QListWidget()
        self.right_layout.addWidget(self.table_list)
        self.main_layout.addLayout(self.right_layout)

        #make a button to delete the selected table
        self.delete_table_button = QPushButton("Delete Table", self)
        self.delete_table_button.clicked.connect(self.delete_table)
        self.right_layout.addWidget(self.delete_table_button)

    def new_file(self):
        self.file_path = QFileDialog.getSaveFileName(self, "Save File", filter="Prism Files (*.pzfx)")
        self.file_path = self.file_path[0]
        self.prism_writer = prism_writer.PrismFile()
        self.save_file_button.clicked.connect(self.save_file)
        self.save_file_button.setDisabled(False)
        self.prism_title.setText(f"Prism Writer - {self.file_path}")
        self.table_list.clear()

    def save_file(self):
        self.prism_writer.save(self.file_path)

    def open_csv(self):
        #or open xlsx
        self.csv_path = QFileDialog.getOpenFileName(self, "Open CSV", filter="Excel Files (*.csv, *.xlsx)")
        self.csv_path = self.csv_path[0]
        self.df = pd.read_csv(self.csv_path) if self.csv_path.endswith('.csv') else pd.read_excel(self.csv_path)
        #in this case, we are just needing the indexs and the columns
        self.rows = self.df.index.values
        self.columns = self.df.columns.values

        #clear the list widgets
        self.main_group_list.clear()
        self.sub_group_list.clear()
        self.row_group_list.clear()
        

        #populate the main group list with the columns
        self.main_group_list.addItems([f'[COL] - {x}' for x in self.columns])
        #populate the sub group list with the columns
        self.sub_group_list.addItems([f'[COL] - {x}' for x in self.columns])
        #populate the row group list with the rows
        self.row_group_list.addItems([f'[COL] - {x}' for x in self.columns])


    def create_group_table(self, _):
        #get the args and pass them to the prism writer
        #find what is selected in each list
        main_group = self.main_group_list.selectedItems()
        sub_group = self.sub_group_list.selectedItems()
        row_group = self.row_group_list.selectedItems()
        #get the indexes
        main_group = [x.text().split(' - ')[1] for x in main_group]
        sub_group = [x.text().split(' - ')[1] for x in sub_group]
        row_group = [x.text().split(' - ')[1] for x in row_group]
        #add the group to the prism writer
        if len(sub_group) >1:
            sub_group_cols = copy.copy(sub_group)
            sub_group = None
        elif len(sub_group) == 1:
            sub_group_cols = None
            sub_group = sub_group[0]
        else:
            sub_group_cols = None
            sub_group = None


        if len(row_group) >1:
            row_group_cols = copy.copy(row_group)
            row_group = None
        elif len(row_group) == 1:
            row_group_cols = None
            row_group = row_group[0]
        else:
            row_group_cols = None
            row_group = None

        self.prism_writer.make_group_table(self.group_table_name.text(), self.df, main_group, cols=None, 
                                           subgroupcols=sub_group_cols, rowgroupcols=row_group_cols, subgroupby=sub_group, rowgroupby=row_group)

        self.table_list.addItem(f"Group Table - {self.group_table_name.text()}")

    def delete_table(self):
        #get the selected table
        table = self.table_list.selectedItems()
        table = table[0].text().split(' - ')[1]
        self.prism_writer.delete_table(table)
        self.table_list.takeItem(self.table_list.currentRow())

if __name__ == '__main__':
    app = QApplication([])
    ex = PrismWriterGUI()
    app.exec_()



        