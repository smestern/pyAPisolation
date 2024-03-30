import numpy 
import os
import sys
import time
import argparse
import run_output_to_web
import build_database
import dash_folder_app
import PySide2.QtWidgets as QtWidgets
import pandas as pd

class WebVizConfigGenerator(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.backend = 'dynamic'
        self.data_folder = None
        self.data_df = None

    def init_ui(self):
        #should have a layout with a file dialog to select the data folder
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)
        self.file_dialog = QtWidgets.QFileDialog()
        self.file_dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        self.layout.addWidget(self.file_dialog)
        self.file_dialog.fileSelected.connect(self.set_data_folder)
        self.backend_select = QtWidgets.QComboBox()
        self.backend_select.addItem('static')
        self.backend_select.addItem('dynamic')
        self.backend_select.addItem('dash')
        self.layout.addWidget(self.backend_select)

        #an empty dropdown to select the column for filename, populated after the data folder is selected
        self.filename_select = QtWidgets.QComboBox()
        self.layout.addWidget(self.filename_select) 
        self.filename_select.currentTextChanged.connect(self.set_filename_column)

        #a button to run the web viz
        self.run_button = QtWidgets.QPushButton('Run Web Viz')
        self.run_button.clicked.connect(self.run_web_viz)
        self.layout.addWidget(self.run_button)

    def set_data_folder(self, folder):
        self.data_folder = folder
        self.data_df = build_database.build_database(folder)
        self.filename_select.clear()
        self.filename_select.addItems(self.data_df.columns)

    def set_filename_column(self, column):
        self.filename_column = column
    
    def run_web_viz(self):
        run_output_to_web.main(database_file=self.data_df, static=(self.backend=='static'))
        if self.backend == 'dash':
            app = dash_folder_app.run_app(self.data_df)
            return app
        

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    w = WebVizConfigGenerator()
    w.show()
    sys.exit(app.exec_())



