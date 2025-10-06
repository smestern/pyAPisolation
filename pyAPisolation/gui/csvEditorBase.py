# -*- coding: utf-8 -*-

################################################################################
## CSV/Excel Editor Base UI
## Simplified version based on databaseBuilderBase without tsDatabase dependencies
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_csvEditorBase(object):
    def setupUi(self, csvEditorBase):
        if not csvEditorBase.objectName():
            csvEditorBase.setObjectName(u"csvEditorBase")
        csvEditorBase.resize(1200, 700)
        self.centralwidget = QWidget(csvEditorBase)
        self.centralwidget.setObjectName(u"centralwidget")
        
        # Main grid layout with column stretch factors for 75%-25% split
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setColumnStretch(0, 75)  # Left column (75% width)
        self.gridLayout.setColumnStretch(1, 25)  # Right column (25% width)

        # Left column layout (75% width)
        self.leftColumnLayout = QVBoxLayout()
        self.leftColumnLayout.setObjectName(u"leftColumnLayout")
        
        # Table section (main content area)
        self.tableContainer = QVBoxLayout()
        self.tableContainer.setObjectName(u"tableContainer")
        
        self.tableLabel = QLabel(self.centralwidget)
        self.tableLabel.setObjectName(u"tableLabel")
        self.tableContainer.addWidget(self.tableLabel)

        self.tableFrame = QFrame(self.centralwidget)
        self.tableFrame.setObjectName(u"tableFrame")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(75)
        sizePolicy.setVerticalStretch(80)
        sizePolicy.setHeightForWidth(self.tableFrame.sizePolicy().hasHeightForWidth())
        self.tableFrame.setSizePolicy(sizePolicy)
        self.tableFrame.setFrameShape(QFrame.StyledPanel)
        self.tableFrame.setFrameShadow(QFrame.Raised)
        self.tableFrame.setAcceptDrops(True)
        
        self.tableContainer.addWidget(self.tableFrame)
        
        # Button layout for table operations
        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.setObjectName(u"buttonLayout")
        
        self.addRowButton = QPushButton(self.centralwidget)
        self.addRowButton.setObjectName(u"addRowButton")
        self.buttonLayout.addWidget(self.addRowButton)

        self.addColumnButton = QPushButton(self.centralwidget)
        self.addColumnButton.setObjectName(u"addColumnButton")
        self.buttonLayout.addWidget(self.addColumnButton)
        
        self.deleteRowButton = QPushButton(self.centralwidget)
        self.deleteRowButton.setObjectName(u"deleteRowButton")
        self.buttonLayout.addWidget(self.deleteRowButton)
        
        self.deleteColumnButton = QPushButton(self.centralwidget)
        self.deleteColumnButton.setObjectName(u"deleteColumnButton")
        self.buttonLayout.addWidget(self.deleteColumnButton)
        
        self.tableContainer.addLayout(self.buttonLayout)
        
        self.leftColumnLayout.addLayout(self.tableContainer, 80)  # 80% of left column height

        # Status section (20% of left column height)
        self.statusGroup = QGroupBox(self.centralwidget)
        self.statusGroup.setObjectName(u"statusGroup")
        self.statusGroup.setMinimumSize(QSize(0, 120))
        statusSizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        statusSizePolicy.setHorizontalStretch(75)
        statusSizePolicy.setVerticalStretch(20)
        self.statusGroup.setSizePolicy(statusSizePolicy)
        
        # Status text area
        self.statusLayout = QVBoxLayout(self.statusGroup)
        self.statusText = QTextEdit(self.statusGroup)
        self.statusText.setMaximumHeight(80)
        self.statusText.setReadOnly(True)
        self.statusLayout.addWidget(self.statusText)
        
        self.leftColumnLayout.addWidget(self.statusGroup, 20)  # 20% of left column height

        # Add left column to main grid
        self.gridLayout.addLayout(self.leftColumnLayout, 0, 0, 1, 1)

        # Right column layout (25% width) - File Browser
        self.rightColumnLayout = QVBoxLayout()
        self.rightColumnLayout.setObjectName(u"rightColumnLayout")
        
        self.fileBrowserLabel = QLabel(self.centralwidget)
        self.fileBrowserLabel.setObjectName(u"fileBrowserLabel")
        self.rightColumnLayout.addWidget(self.fileBrowserLabel)

        self.fileBrowserFrame = QFrame(self.centralwidget)
        self.fileBrowserFrame.setObjectName(u"fileBrowserFrame")
        fileBrowserSizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        fileBrowserSizePolicy.setHorizontalStretch(25)
        fileBrowserSizePolicy.setVerticalStretch(0)
        fileBrowserSizePolicy.setHeightForWidth(self.fileBrowserFrame.sizePolicy().hasHeightForWidth())
        self.fileBrowserFrame.setSizePolicy(fileBrowserSizePolicy)
        self.fileBrowserFrame.setFrameShape(QFrame.StyledPanel)
        self.fileBrowserFrame.setFrameShadow(QFrame.Raised)
        self.rightColumnLayout.addWidget(self.fileBrowserFrame)

        # Add right column to main grid
        self.gridLayout.addLayout(self.rightColumnLayout, 0, 1, 1, 1)

        csvEditorBase.setCentralWidget(self.centralwidget)
        
        # Menu bar
        self.menubar = QMenuBar(csvEditorBase)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1200, 21))
        
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        
        self.menuEdit = QMenu(self.menubar)
        self.menuEdit.setObjectName(u"menuEdit")
        
        csvEditorBase.setMenuBar(self.menubar)
        
        # Status bar
        self.statusbar = QStatusBar(csvEditorBase)
        self.statusbar.setObjectName(u"statusbar")
        csvEditorBase.setStatusBar(self.statusbar)

        # Add actions to menu
        self.actionOpen = QAction(csvEditorBase)
        self.actionOpen.setObjectName(u"actionOpen")
        
        self.actionSave = QAction(csvEditorBase)
        self.actionSave.setObjectName(u"actionSave")
        
        self.actionSaveAs = QAction(csvEditorBase)
        self.actionSaveAs.setObjectName(u"actionSaveAs")
        
        self.actionNew = QAction(csvEditorBase)
        self.actionNew.setObjectName(u"actionNew")
        
        self.actionExit = QAction(csvEditorBase)
        self.actionExit.setObjectName(u"actionExit")
        
        self.menuFile.addAction(self.actionNew)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionSaveAs)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuEdit.menuAction())

        self.retranslateUi(csvEditorBase)

        QMetaObject.connectSlotsByName(csvEditorBase)
    # setupUi

    def retranslateUi(self, csvEditorBase):
        csvEditorBase.setWindowTitle(QCoreApplication.translate("csvEditorBase", u"CSV/Excel Editor - pyAPisolation", None))
        self.tableLabel.setText(QCoreApplication.translate("csvEditorBase", u"Spreadsheet Editor (Drag files into cells)", None))
        self.addRowButton.setText(QCoreApplication.translate("csvEditorBase", u"Add Row", None))
        self.addColumnButton.setText(QCoreApplication.translate("csvEditorBase", u"Add Column", None))
        self.deleteRowButton.setText(QCoreApplication.translate("csvEditorBase", u"Delete Row", None))
        self.deleteColumnButton.setText(QCoreApplication.translate("csvEditorBase", u"Delete Column", None))
        self.statusGroup.setTitle(QCoreApplication.translate("csvEditorBase", u"Status & Information", None))
        self.fileBrowserLabel.setText(QCoreApplication.translate("csvEditorBase", u"File Browser", None))
        self.menuFile.setTitle(QCoreApplication.translate("csvEditorBase", u"File", None))
        self.menuEdit.setTitle(QCoreApplication.translate("csvEditorBase", u"Edit", None))
        self.actionOpen.setText(QCoreApplication.translate("csvEditorBase", u"Open", None))
        self.actionSave.setText(QCoreApplication.translate("csvEditorBase", u"Save", None))
        self.actionSaveAs.setText(QCoreApplication.translate("csvEditorBase", u"Save As...", None))
        self.actionNew.setText(QCoreApplication.translate("csvEditorBase", u"New", None))
        self.actionExit.setText(QCoreApplication.translate("csvEditorBase", u"Exit", None))
    # retranslateUi