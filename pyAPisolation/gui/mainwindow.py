# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'mainwindowMDI.ui'
##
## Created by: Qt User Interface Compiler version 5.15.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1721, 1083)
        MainWindow.setToolTipDuration(-3)
        MainWindow.setDocumentMode(False)
        MainWindow.setDockNestingEnabled(False)
        self.actionEnable_Parallel = QAction(MainWindow)
        self.actionEnable_Parallel.setObjectName(u"actionEnable_Parallel")
        self.actionEnable_Parallel.setCheckable(True)
        self.actionShow_Rejected_Spikes = QAction(MainWindow)
        self.actionShow_Rejected_Spikes.setObjectName(u"actionShow_Rejected_Spikes")
        self.actionShow_Rejected_Spikes.setCheckable(True)
        self.actionOpen_Folder = QAction(MainWindow)
        self.actionOpen_Folder.setObjectName(u"actionOpen_Folder")
        self.actionOrganize_Abf = QAction(MainWindow)
        self.actionOrganize_Abf.setObjectName(u"actionOrganize_Abf")
        self.actionOpen_Results = QAction(MainWindow)
        self.actionOpen_Results.setObjectName(u"actionOpen_Results")
        self.actionExit = QAction(MainWindow)
        self.actionExit.setObjectName(u"actionExit")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.centralwidget.setToolTipDuration(-3)
        self.verticalLayout_7 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_7.setSpacing(3)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.verticalLayout_7.setContentsMargins(-1, -1, 3, 8)
        self.mdiArea = QMdiArea(self.centralwidget)
        self.mdiArea.setObjectName(u"mdiArea")
        sizePolicy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.mdiArea.sizePolicy().hasHeightForWidth())
        self.mdiArea.setSizePolicy(sizePolicy)
        self.mdiArea.setMinimumSize(QSize(800, 600))
        self.mdiArea.setMaximumSize(QSize(9999999, 999999))
        self.mdiArea.setFrameShape(QFrame.Panel)
        self.mdiArea.setFrameShadow(QFrame.Raised)
        self.mdiArea.setLineWidth(-1)
        self.mdiArea.setMidLineWidth(0)
        self.mdiArea.setViewMode(QMdiArea.SubWindowView)
        self.mdiArea.setDocumentMode(False)
        self.mdiArea.setTabsClosable(False)
        self.mdiArea.setTabsMovable(False)
        self.subwindow = QWidget()
        self.subwindow.setObjectName(u"subwindow")
        self.gridLayout_2 = QGridLayout(self.subwindow)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.gridLayout_2.setContentsMargins(-1, -1, -1, 6)
        self.analysis_set = QGroupBox(self.subwindow)
        self.analysis_set.setObjectName(u"analysis_set")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.analysis_set.sizePolicy().hasHeightForWidth())
        self.analysis_set.setSizePolicy(sizePolicy1)
        self.analysis_set.setMinimumSize(QSize(400, 0))
        self.analysis_set.setMaximumSize(QSize(600, 16777215))
        self.gridLayout = QGridLayout(self.analysis_set)
        self.gridLayout.setObjectName(u"gridLayout")
        self.tabWidget = QTabWidget(self.analysis_set)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tabWidget.setUsesScrollButtons(False)
        self.runspikefinder = QWidget()
        self.runspikefinder.setObjectName(u"runspikefinder")
        self.verticalLayout_2 = QVBoxLayout(self.runspikefinder)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setSizeConstraint(QLayout.SetFixedSize)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalLayout.setSizeConstraint(QLayout.SetMaximumSize)
        self.label = QLabel(self.runspikefinder)
        self.label.setObjectName(u"label")
        sizePolicy2 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Minimum)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy2)
        self.label.setMinimumSize(QSize(155, 0))

        self.horizontalLayout.addWidget(self.label)

        self.dvdt_thres = QDoubleSpinBox(self.runspikefinder)
        self.dvdt_thres.setObjectName(u"dvdt_thres")
        self.dvdt_thres.setSingleStep(0.500000000000000)
        self.dvdt_thres.setValue(7.000000000000000)

        self.horizontalLayout.addWidget(self.dvdt_thres)


        self.verticalLayout_3.addLayout(self.horizontalLayout)

        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_3 = QLabel(self.runspikefinder)
        self.label_3.setObjectName(u"label_3")
        sizePolicy3 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        sizePolicy3.setHorizontalStretch(0)
        sizePolicy3.setVerticalStretch(0)
        sizePolicy3.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy3)
        self.label_3.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_3.addWidget(self.label_3)

        self.t_to_p_time = QDoubleSpinBox(self.runspikefinder)
        self.t_to_p_time.setObjectName(u"t_to_p_time")
        self.t_to_p_time.setValue(5.000000000000000)

        self.horizontalLayout_3.addWidget(self.t_to_p_time)


        self.verticalLayout_3.addLayout(self.horizontalLayout_3)

        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_4 = QLabel(self.runspikefinder)
        self.label_4.setObjectName(u"label_4")
        sizePolicy4 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy4.setHorizontalStretch(150)
        sizePolicy4.setVerticalStretch(0)
        sizePolicy4.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy4)
        self.label_4.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_4.addWidget(self.label_4)

        self.t_to_p_height = QDoubleSpinBox(self.runspikefinder)
        self.t_to_p_height.setObjectName(u"t_to_p_height")
        self.t_to_p_height.setValue(2.000000000000000)

        self.horizontalLayout_4.addWidget(self.t_to_p_height)


        self.verticalLayout_3.addLayout(self.horizontalLayout_4)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_6 = QLabel(self.runspikefinder)
        self.label_6.setObjectName(u"label_6")
        sizePolicy4.setHeightForWidth(self.label_6.sizePolicy().hasHeightForWidth())
        self.label_6.setSizePolicy(sizePolicy4)
        self.label_6.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_6.addWidget(self.label_6)

        self.min_peak = QDoubleSpinBox(self.runspikefinder)
        self.min_peak.setObjectName(u"min_peak")
        self.min_peak.setMinimum(-99.000000000000000)
        self.min_peak.setValue(-10.000000000000000)

        self.horizontalLayout_6.addWidget(self.min_peak)


        self.verticalLayout_3.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setSizeConstraint(QLayout.SetMaximumSize)
        self.bstim = QCheckBox(self.runspikefinder)
        self.bstim.setObjectName(u"bstim")

        self.horizontalLayout_7.addWidget(self.bstim)


        self.verticalLayout_3.addLayout(self.horizontalLayout_7)

        self.horizontalLayout_8 = QHBoxLayout()
        self.horizontalLayout_8.setObjectName(u"horizontalLayout_8")
        self.horizontalLayout_8.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_7 = QLabel(self.runspikefinder)
        self.label_7.setObjectName(u"label_7")
        sizePolicy5 = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        sizePolicy5.setHorizontalStretch(150)
        sizePolicy5.setVerticalStretch(0)
        sizePolicy5.setHeightForWidth(self.label_7.sizePolicy().hasHeightForWidth())
        self.label_7.setSizePolicy(sizePolicy5)
        self.label_7.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_8.addWidget(self.label_7)

        self.start = QDoubleSpinBox(self.runspikefinder)
        self.start.setObjectName(u"start")
        self.start.setDecimals(4)
        self.start.setMaximum(99999999.000000000000000)

        self.horizontalLayout_8.addWidget(self.start)


        self.verticalLayout_3.addLayout(self.horizontalLayout_8)

        self.horizontalLayout_9 = QHBoxLayout()
        self.horizontalLayout_9.setObjectName(u"horizontalLayout_9")
        self.horizontalLayout_9.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_8 = QLabel(self.runspikefinder)
        self.label_8.setObjectName(u"label_8")
        sizePolicy5.setHeightForWidth(self.label_8.sizePolicy().hasHeightForWidth())
        self.label_8.setSizePolicy(sizePolicy5)
        self.label_8.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_9.addWidget(self.label_8)

        self.end_time = QDoubleSpinBox(self.runspikefinder)
        self.end_time.setObjectName(u"end_time")
        self.end_time.setDecimals(4)
        self.end_time.setMaximum(99999999.000000000000000)

        self.horizontalLayout_9.addWidget(self.end_time)


        self.verticalLayout_3.addLayout(self.horizontalLayout_9)

        self.horizontalLayout_10 = QHBoxLayout()
        self.horizontalLayout_10.setObjectName(u"horizontalLayout_10")
        self.horizontalLayout_10.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_2 = QLabel(self.runspikefinder)
        self.label_2.setObjectName(u"label_2")
        sizePolicy4.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy4)
        self.label_2.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_10.addWidget(self.label_2)

        self.bessel_filt = QDoubleSpinBox(self.runspikefinder)
        self.bessel_filt.setObjectName(u"bessel_filt")
        self.bessel_filt.setMinimum(-1.000000000000000)
        self.bessel_filt.setMaximum(9000000.000000000000000)
        self.bessel_filt.setSingleStep(500.000000000000000)
        self.bessel_filt.setValue(9999.000000000000000)

        self.horizontalLayout_10.addWidget(self.bessel_filt)


        self.verticalLayout_3.addLayout(self.horizontalLayout_10)

        self.horizontalLayout_12 = QHBoxLayout()
        self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
        self.horizontalLayout_12.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_5 = QLabel(self.runspikefinder)
        self.label_5.setObjectName(u"label_5")
        sizePolicy4.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy4)
        self.label_5.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_12.addWidget(self.label_5)

        self.thres_percent = QDoubleSpinBox(self.runspikefinder)
        self.thres_percent.setObjectName(u"thres_percent")
        self.thres_percent.setDecimals(3)
        self.thres_percent.setMinimum(-1.000000000000000)
        self.thres_percent.setMaximum(1.000000000000000)
        self.thres_percent.setSingleStep(500.000000000000000)
        self.thres_percent.setValue(0.200000000000000)

        self.horizontalLayout_12.addWidget(self.thres_percent)


        self.verticalLayout_3.addLayout(self.horizontalLayout_12)

        self.horizontalLayout_17 = QHBoxLayout()
        self.horizontalLayout_17.setSpacing(6)
        self.horizontalLayout_17.setObjectName(u"horizontalLayout_17")
        self.horizontalLayout_17.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_11 = QLabel(self.runspikefinder)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setEnabled(True)
        sizePolicy6 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Preferred)
        sizePolicy6.setHorizontalStretch(0)
        sizePolicy6.setVerticalStretch(0)
        sizePolicy6.setHeightForWidth(self.label_11.sizePolicy().hasHeightForWidth())
        self.label_11.setSizePolicy(sizePolicy6)

        self.horizontalLayout_17.addWidget(self.label_11)

        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.spikeFinder = QCheckBox(self.runspikefinder)
        self.spikeFinder.setObjectName(u"spikeFinder")
        sizePolicy7 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.Fixed)
        sizePolicy7.setHorizontalStretch(0)
        sizePolicy7.setVerticalStretch(0)
        sizePolicy7.setHeightForWidth(self.spikeFinder.sizePolicy().hasHeightForWidth())
        self.spikeFinder.setSizePolicy(sizePolicy7)
        self.spikeFinder.setChecked(True)

        self.verticalLayout_5.addWidget(self.spikeFinder)

        self.rawSpike = QCheckBox(self.runspikefinder)
        self.rawSpike.setObjectName(u"rawSpike")
        sizePolicy7.setHeightForWidth(self.rawSpike.sizePolicy().hasHeightForWidth())
        self.rawSpike.setSizePolicy(sizePolicy7)

        self.verticalLayout_5.addWidget(self.rawSpike)

        self.runningBin = QCheckBox(self.runspikefinder)
        self.runningBin.setObjectName(u"runningBin")
        sizePolicy7.setHeightForWidth(self.runningBin.sizePolicy().hasHeightForWidth())
        self.runningBin.setSizePolicy(sizePolicy7)

        self.verticalLayout_5.addWidget(self.runningBin)


        self.horizontalLayout_17.addLayout(self.verticalLayout_5)


        self.verticalLayout_3.addLayout(self.horizontalLayout_17)


        self.verticalLayout_2.addLayout(self.verticalLayout_3)

        self.tabWidget.addTab(self.runspikefinder, "")
        self.runcmcalc = QWidget()
        self.runcmcalc.setObjectName(u"runcmcalc")
        self.runcmcalc.setEnabled(True)
        self.runcmcalc.setMaximumSize(QSize(16777215, 16777212))
        self.verticalLayout_4 = QVBoxLayout(self.runcmcalc)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setSizeConstraint(QLayout.SetFixedSize)
        self.horizontalLayout_11 = QHBoxLayout()
        self.horizontalLayout_11.setObjectName(u"horizontalLayout_11")
        self.horizontalLayout_11.setSizeConstraint(QLayout.SetFixedSize)
        self.label_10 = QLabel(self.runcmcalc)
        self.label_10.setObjectName(u"label_10")
        sizePolicy2.setHeightForWidth(self.label_10.sizePolicy().hasHeightForWidth())
        self.label_10.setSizePolicy(sizePolicy2)
        self.label_10.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_11.addWidget(self.label_10)

        self.stimPer = QDoubleSpinBox(self.runcmcalc)
        self.stimPer.setObjectName(u"stimPer")
        sizePolicy8 = QSizePolicy(QSizePolicy.Minimum, QSizePolicy.Fixed)
        sizePolicy8.setHorizontalStretch(0)
        sizePolicy8.setVerticalStretch(0)
        sizePolicy8.setHeightForWidth(self.stimPer.sizePolicy().hasHeightForWidth())
        self.stimPer.setSizePolicy(sizePolicy8)
        self.stimPer.setMinimumSize(QSize(0, 0))
        self.stimPer.setWrapping(False)
        self.stimPer.setFrame(True)
        self.stimPer.setValue(50.000000000000000)

        self.horizontalLayout_11.addWidget(self.stimPer)


        self.verticalLayout_6.addLayout(self.horizontalLayout_11)

        self.horizontalLayout_15 = QHBoxLayout()
        self.horizontalLayout_15.setObjectName(u"horizontalLayout_15")
        self.horizontalLayout_15.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_13 = QLabel(self.runcmcalc)
        self.label_13.setObjectName(u"label_13")
        sizePolicy9 = QSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        sizePolicy9.setHorizontalStretch(0)
        sizePolicy9.setVerticalStretch(0)
        sizePolicy9.setHeightForWidth(self.label_13.sizePolicy().hasHeightForWidth())
        self.label_13.setSizePolicy(sizePolicy9)
        self.label_13.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_15.addWidget(self.label_13)

        self.endCM = QDoubleSpinBox(self.runcmcalc)
        self.endCM.setObjectName(u"endCM")
        self.endCM.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_15.addWidget(self.endCM)


        self.verticalLayout_6.addLayout(self.horizontalLayout_15)

        self.horizontalLayout_13 = QHBoxLayout()
        self.horizontalLayout_13.setObjectName(u"horizontalLayout_13")
        self.horizontalLayout_13.setSizeConstraint(QLayout.SetMaximumSize)
        self.bstim_2 = QCheckBox(self.runcmcalc)
        self.bstim_2.setObjectName(u"bstim_2")

        self.horizontalLayout_13.addWidget(self.bstim_2)


        self.verticalLayout_6.addLayout(self.horizontalLayout_13)

        self.horizontalLayout_14 = QHBoxLayout()
        self.horizontalLayout_14.setObjectName(u"horizontalLayout_14")
        self.horizontalLayout_14.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_12 = QLabel(self.runcmcalc)
        self.label_12.setObjectName(u"label_12")
        sizePolicy9.setHeightForWidth(self.label_12.sizePolicy().hasHeightForWidth())
        self.label_12.setSizePolicy(sizePolicy9)
        self.label_12.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_14.addWidget(self.label_12)

        self.startCM = QDoubleSpinBox(self.runcmcalc)
        self.startCM.setObjectName(u"startCM")
        sizePolicy7.setHeightForWidth(self.startCM.sizePolicy().hasHeightForWidth())
        self.startCM.setSizePolicy(sizePolicy7)
        self.startCM.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_14.addWidget(self.startCM)


        self.verticalLayout_6.addLayout(self.horizontalLayout_14)

        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_9 = QLabel(self.runcmcalc)
        self.label_9.setObjectName(u"label_9")
        sizePolicy2.setHeightForWidth(self.label_9.sizePolicy().hasHeightForWidth())
        self.label_9.setSizePolicy(sizePolicy2)
        self.label_9.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_5.addWidget(self.label_9)

        self.subthresSweeps = QLineEdit(self.runcmcalc)
        self.subthresSweeps.setObjectName(u"subthresSweeps")
        sizePolicy7.setHeightForWidth(self.subthresSweeps.sizePolicy().hasHeightForWidth())
        self.subthresSweeps.setSizePolicy(sizePolicy7)
        self.subthresSweeps.setMinimumSize(QSize(100, 0))

        self.horizontalLayout_5.addWidget(self.subthresSweeps)


        self.verticalLayout_6.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_16 = QHBoxLayout()
        self.horizontalLayout_16.setObjectName(u"horizontalLayout_16")
        self.horizontalLayout_16.setSizeConstraint(QLayout.SetMaximumSize)
        self.label_14 = QLabel(self.runcmcalc)
        self.label_14.setObjectName(u"label_14")
        sizePolicy9.setHeightForWidth(self.label_14.sizePolicy().hasHeightForWidth())
        self.label_14.setSizePolicy(sizePolicy9)
        self.label_14.setMinimumSize(QSize(155, 0))

        self.horizontalLayout_16.addWidget(self.label_14)

        self.bessel_filt_cm = QDoubleSpinBox(self.runcmcalc)
        self.bessel_filt_cm.setObjectName(u"bessel_filt_cm")
        sizePolicy7.setHeightForWidth(self.bessel_filt_cm.sizePolicy().hasHeightForWidth())
        self.bessel_filt_cm.setSizePolicy(sizePolicy7)
        self.bessel_filt_cm.setMinimumSize(QSize(0, 25))
        self.bessel_filt_cm.setMinimum(-1.000000000000000)
        self.bessel_filt_cm.setMaximum(9000000.000000000000000)
        self.bessel_filt_cm.setSingleStep(500.000000000000000)
        self.bessel_filt_cm.setValue(4999.000000000000000)

        self.horizontalLayout_16.addWidget(self.bessel_filt_cm)


        self.verticalLayout_6.addLayout(self.horizontalLayout_16)


        self.verticalLayout_4.addLayout(self.verticalLayout_6)

        self.tabWidget.addTab(self.runcmcalc, "")

        self.gridLayout.addWidget(self.tabWidget, 1, 0, 1, 1)


        self.gridLayout_2.addWidget(self.analysis_set, 0, 0, 1, 1)

        self.frame_2 = QFrame(self.subwindow)
        self.frame_2.setObjectName(u"frame_2")
        sizePolicy10 = QSizePolicy(QSizePolicy.Maximum, QSizePolicy.Minimum)
        sizePolicy10.setHorizontalStretch(0)
        sizePolicy10.setVerticalStretch(0)
        sizePolicy10.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy10)
        self.frame_2.setMinimumSize(QSize(0, 150))
        self.frame_2.setMaximumSize(QSize(600, 16777215))
        self.frame_2.setFrameShape(QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.frame_2)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.protocol_selector = QComboBox(self.frame_2)
        self.protocol_selector.setObjectName(u"protocol_selector")

        self.verticalLayout.addWidget(self.protocol_selector)

        self.outputTag = QLineEdit(self.frame_2)
        self.outputTag.setObjectName(u"outputTag")

        self.verticalLayout.addWidget(self.outputTag)

        self.run_analysis = QPushButton(self.frame_2)
        self.run_analysis.setObjectName(u"run_analysis")

        self.verticalLayout.addWidget(self.run_analysis)

        self.saveCur = QPushButton(self.frame_2)
        self.saveCur.setObjectName(u"saveCur")

        self.verticalLayout.addWidget(self.saveCur)


        self.gridLayout_2.addWidget(self.frame_2, 1, 0, 1, 1)

        self.mdiArea.addSubWindow(self.subwindow)
        self.subwindow_2 = QWidget()
        self.subwindow_2.setObjectName(u"subwindow_2")
        sizePolicy11 = QSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
        sizePolicy11.setHorizontalStretch(0)
        sizePolicy11.setVerticalStretch(0)
        sizePolicy11.setHeightForWidth(self.subwindow_2.sizePolicy().hasHeightForWidth())
        self.subwindow_2.setSizePolicy(sizePolicy11)
        self.subwindow_2.setMinimumSize(QSize(400, 300))
        self.subwindow_2.setMaximumSize(QSize(16777215, 16777215))
        self.subwindow_2.setToolTipDuration(5)
        self.verticalLayout_9 = QVBoxLayout(self.subwindow_2)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.refresh_plot = QPushButton(self.subwindow_2)
        self.refresh_plot.setObjectName(u"refresh_plot")
        font = QFont()
        font.setPointSize(14)
        self.refresh_plot.setFont(font)

        self.verticalLayout_9.addWidget(self.refresh_plot)

        self.mainplot = QWidget(self.subwindow_2)
        self.mainplot.setObjectName(u"mainplot")

        self.verticalLayout_9.addWidget(self.mainplot)

        self.sweep_selector = QWidget(self.subwindow_2)
        self.sweep_selector.setObjectName(u"sweep_selector")
        self.sweep_selector.setMinimumSize(QSize(0, 50))
        self.sweep_selector.setMaximumSize(QSize(16777215, 50))

        self.verticalLayout_9.addWidget(self.sweep_selector)

        self.mdiArea.addSubWindow(self.subwindow_2)
        self.subwindow_3 = QWidget()
        self.subwindow_3.setObjectName(u"subwindow_3")
        self.verticalLayout_10 = QVBoxLayout(self.subwindow_3)
        self.verticalLayout_10.setObjectName(u"verticalLayout_10")
        self.folder_select = QPushButton(self.subwindow_3)
        self.folder_select.setObjectName(u"folder_select")

        self.verticalLayout_10.addWidget(self.folder_select)

        self.file_list = QListWidget(self.subwindow_3)
        self.file_list.setObjectName(u"file_list")
        sizePolicy12 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy12.setHorizontalStretch(0)
        sizePolicy12.setVerticalStretch(0)
        sizePolicy12.setHeightForWidth(self.file_list.sizePolicy().hasHeightForWidth())
        self.file_list.setSizePolicy(sizePolicy12)
        self.file_list.setMaximumSize(QSize(16777215, 16777215))

        self.verticalLayout_10.addWidget(self.file_list)

        self.mdiArea.addSubWindow(self.subwindow_3)
        self.resultsWindow = QWidget()
        self.resultsWindow.setObjectName(u"resultsWindow")
        self.formLayout = QFormLayout(self.resultsWindow)
        self.formLayout.setObjectName(u"formLayout")
        self.formLayout.setContentsMargins(-1, 13, -1, -1)
        self.resultsTable = QTableView(self.resultsWindow)
        self.resultsTable.setObjectName(u"resultsTable")
        self.resultsTable.setToolTipDuration(-2)
        self.resultsTable.setMidLineWidth(-1)

        self.formLayout.setWidget(0, QFormLayout.SpanningRole, self.resultsTable)

        self.mdiArea.addSubWindow(self.resultsWindow)

        self.verticalLayout_7.addWidget(self.mdiArea)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1721, 21))
        self.menuFile = QMenu(self.menubar)
        self.menuFile.setObjectName(u"menuFile")
        self.menuTools = QMenu(self.menubar)
        self.menuTools.setObjectName(u"menuTools")
        self.menuSettings = QMenu(self.menubar)
        self.menuSettings.setObjectName(u"menuSettings")
        self.menuDebug = QMenu(self.menuSettings)
        self.menuDebug.setObjectName(u"menuDebug")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuTools.menuAction())
        self.menubar.addAction(self.menuSettings.menuAction())
        self.menuFile.addAction(self.actionOpen_Folder)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen_Results)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionExit)
        self.menuTools.addAction(self.actionOrganize_Abf)
        self.menuSettings.addAction(self.actionEnable_Parallel)
        self.menuSettings.addAction(self.menuDebug.menuAction())
        self.menuDebug.addAction(self.actionShow_Rejected_Spikes)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)


        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Spike Finder", None))
        self.actionEnable_Parallel.setText(QCoreApplication.translate("MainWindow", u"Enable Parallel", None))
        self.actionShow_Rejected_Spikes.setText(QCoreApplication.translate("MainWindow", u"Show Rejected Spikes", None))
        self.actionOpen_Folder.setText(QCoreApplication.translate("MainWindow", u"Open Folder", None))
        self.actionOrganize_Abf.setText(QCoreApplication.translate("MainWindow", u"Organize Abf", None))
        self.actionOpen_Results.setText(QCoreApplication.translate("MainWindow", u"Open Results", None))
        self.actionExit.setText(QCoreApplication.translate("MainWindow", u"Exit", None))
        self.subwindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Analysis Settings", None))
        self.analysis_set.setTitle(QCoreApplication.translate("MainWindow", u"Analysis Settings", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"dV/dT (mV/ms)", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"max threshold-to-peak time (ms)", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"min thres-to-peak height (mV)", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"min peak (mV)", None))
        self.bstim.setText(QCoreApplication.translate("MainWindow", u"Find Stim Time Automatically", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Start Search Period (S)", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"End Search Period (s)", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Bessel Filter (Hz)", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Threshold Refine Percent (%)", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Output", None))
        self.spikeFinder.setText(QCoreApplication.translate("MainWindow", u"SpikeFinder Main Sheet", None))
        self.rawSpike.setText(QCoreApplication.translate("MainWindow", u"Raw Spike Data", None))
        self.runningBin.setText(QCoreApplication.translate("MainWindow", u"Running Bin", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.runspikefinder), QCoreApplication.translate("MainWindow", u"run spike finder", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Percentage of Stim to use", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"End Search Period (s)", None))
        self.bstim_2.setText(QCoreApplication.translate("MainWindow", u"Find Stim Time Automatically", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Start Search Period (S)", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Subthreshold Sweeps", None))
        self.label_14.setText(QCoreApplication.translate("MainWindow", u"Bessel Filter (Hz)", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.runcmcalc), QCoreApplication.translate("MainWindow", u"run cm calc", None))
        self.outputTag.setText("")
        self.outputTag.setPlaceholderText(QCoreApplication.translate("MainWindow", u"output_tag", None))
        self.run_analysis.setText(QCoreApplication.translate("MainWindow", u"Analyze Folder", None))
        self.saveCur.setText(QCoreApplication.translate("MainWindow", u"Save Current File Analysis", None))
        self.subwindow_2.setWindowTitle(QCoreApplication.translate("MainWindow", u"Active File Plot", None))
        self.refresh_plot.setText(QCoreApplication.translate("MainWindow", u"\ud83d\udd04 Refresh plot", None))
        self.subwindow_3.setWindowTitle(QCoreApplication.translate("MainWindow", u"File Select", None))
        self.folder_select.setText(QCoreApplication.translate("MainWindow", u"Select a Folder", None))
        self.resultsWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"resultsWindow", None))
        self.menuFile.setTitle(QCoreApplication.translate("MainWindow", u"File", None))
        self.menuTools.setTitle(QCoreApplication.translate("MainWindow", u"Tools", None))
        self.menuSettings.setTitle(QCoreApplication.translate("MainWindow", u"Settings", None))
        self.menuDebug.setTitle(QCoreApplication.translate("MainWindow", u"Debug", None))
    # retranslateUi

