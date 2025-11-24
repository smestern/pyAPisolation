# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'database.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_databaseBuilderBase(object):
    def setupUi(self, databaseBuilderBase):
        if not databaseBuilderBase.objectName():
            databaseBuilderBase.setObjectName(u"databaseBuilderBase")
        databaseBuilderBase.resize(1196, 671)
        self.centralwidget = QWidget(databaseBuilderBase)
        self.centralwidget.setObjectName(u"centralwidget")
        
        # Main grid layout with column stretch factors for 75%-25% split
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(u"gridLayout")
        self.gridLayout.setColumnStretch(0, 75)  # Left column (75% width)
        self.gridLayout.setColumnStretch(1, 25)  # Right column (25% width)

        # Left column layout (75% width)
        self.leftColumnLayout = QVBoxLayout()
        self.leftColumnLayout.setObjectName(u"leftColumnLayout")
        
        # Cell Index section (75% of left column height)
        self.cellIndexContainer = QVBoxLayout()
        self.cellIndexContainer.setObjectName(u"cellIndexContainer")
        
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName(u"label")
        self.cellIndexContainer.addWidget(self.label)

        self.cellIndexFrame = QFrame(self.centralwidget)
        self.cellIndexFrame.setObjectName(u"cellIndexFrame")
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(75)
        sizePolicy.setVerticalStretch(75)
        sizePolicy.setHeightForWidth(self.cellIndexFrame.sizePolicy().hasHeightForWidth())
        self.cellIndexFrame.setSizePolicy(sizePolicy)
        self.cellIndexFrame.setFrameShape(QFrame.StyledPanel)
        self.cellIndexFrame.setFrameShadow(QFrame.Raised)
        self.cellIndexFrame.setAcceptDrops(True)
        
        self.cellIndexContainer.addWidget(self.cellIndexFrame)
        
        # Add button layout
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.addCell = QCommandLinkButton(self.centralwidget)
        self.addCell.setObjectName(u"addCell")
        self.horizontalLayout.addWidget(self.addCell)

        self.addProtocol = QCommandLinkButton(self.centralwidget)
        self.addProtocol.setObjectName(u"addProtocol")
        self.horizontalLayout.addWidget(self.addProtocol)
        
        self.cellIndexContainer.addLayout(self.horizontalLayout)
        
        self.leftColumnLayout.addLayout(self.cellIndexContainer, 75)  # 75% of left column height

        # Metadata section (25% of left column height)
        self.groupBox = QGroupBox(self.centralwidget)
        self.groupBox.setObjectName(u"groupBox")
        self.groupBox.setMinimumSize(QSize(0, 150))
        metadataSizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        metadataSizePolicy.setHorizontalStretch(75)
        metadataSizePolicy.setVerticalStretch(25)
        self.groupBox.setSizePolicy(metadataSizePolicy)
        
        self.leftColumnLayout.addWidget(self.groupBox, 25)  # 25% of left column height

        # Add left column to main grid
        self.gridLayout.addLayout(self.leftColumnLayout, 0, 0, 1, 1)

        # Right column layout (25% width) - File Tree
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.label_2 = QLabel(self.centralwidget)
        self.label_2.setObjectName(u"label_2")
        self.verticalLayout_2.addWidget(self.label_2)

        self.fileTreeFrame = QFrame(self.centralwidget)
        self.fileTreeFrame.setObjectName(u"fileTreeFrame")
        fileTreeSizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        fileTreeSizePolicy.setHorizontalStretch(25)
        fileTreeSizePolicy.setVerticalStretch(0)
        fileTreeSizePolicy.setHeightForWidth(self.fileTreeFrame.sizePolicy().hasHeightForWidth())
        self.fileTreeFrame.setSizePolicy(fileTreeSizePolicy)
        self.fileTreeFrame.setFrameShape(QFrame.StyledPanel)
        self.fileTreeFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2.addWidget(self.fileTreeFrame)

        # Add right column to main grid
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 1, 1, 1)

        databaseBuilderBase.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(databaseBuilderBase)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1196, 21))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        databaseBuilderBase.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(databaseBuilderBase)
        self.statusbar.setObjectName(u"statusbar")
        databaseBuilderBase.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(databaseBuilderBase)

        QMetaObject.connectSlotsByName(databaseBuilderBase)
    # setupUi

    def retranslateUi(self, databaseBuilderBase):
        databaseBuilderBase.setWindowTitle(QCoreApplication.translate("databaseBuilderBase", u"Database Builder - pyAPisolation", None))
        self.label.setText(QCoreApplication.translate("databaseBuilderBase", u"Cell Index (Excel-like view)", None))
        self.addCell.setText(QCoreApplication.translate("databaseBuilderBase", u"Add Cell", None))
        self.addProtocol.setText(QCoreApplication.translate("databaseBuilderBase", u"Add Protocol", None))
        self.groupBox.setTitle(QCoreApplication.translate("databaseBuilderBase", u"Cell Metadata", None))
        self.label_2.setText(QCoreApplication.translate("databaseBuilderBase", u"File Browser", None))
        self.menuFile.setTitle(QCoreApplication.translate("databaseBuilderBase", u"File", None))
    # retranslateUi

