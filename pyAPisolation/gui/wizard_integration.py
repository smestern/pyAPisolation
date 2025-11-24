# -*- coding: utf-8 -*-

"""
Integration helper for Post-hoc Analysis Runner Wizard
Provides functions to integrate the wizard into the main pyAPisolation GUI
"""

from PySide2.QtCore import QObject
from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QAction, QMessageBox, QPushButton
from .postAnalysisRunner import PostAnalysisWizard


def add_analysis_wizard_to_menu(main_window, menu_tools):
    """
    Add the Post-hoc Analysis Wizard to the main application's Tools menu
    
    Args:
        main_window: The main application window
        menu_tools: The Tools menu to add the action to
    """
    # Create action for the wizard
    action_analysis_wizard = QAction(main_window)
    action_analysis_wizard.setObjectName("actionAnalysisWizard")
    action_analysis_wizard.setText("Post-hoc Analysis Wizard...")
    action_analysis_wizard.setToolTip("Run statistical analysis on exported data")
    
    # Connect to wizard launcher
    action_analysis_wizard.triggered.connect(lambda: launch_analysis_wizard(main_window))
    
    # Add to menu
    menu_tools.addSeparator()
    menu_tools.addAction(action_analysis_wizard)
    
    return action_analysis_wizard


def launch_analysis_wizard(parent=None):
    """
    Launch the Post-hoc Analysis Wizard as a modal dialog
    
    Args:
        parent: Parent widget (usually the main window)
    """
    try:
        wizard = PostAnalysisWizard(parent)
        wizard.setModal(True)
        wizard.show()
        return wizard
    except Exception as e:
        QMessageBox.critical(
            parent, 
            "Error", 
            f"Failed to launch Post-hoc Analysis Wizard:\n{str(e)}"
        )
        return None


def create_analysis_wizard_button(parent=None):
    """
    Create a standalone button that launches the analysis wizard
    
    Args:
        parent: Parent widget
        
    Returns:
        QPushButton: Button that launches the wizard
    """
    button = QPushButton("Launch Analysis Wizard", parent)
    button.setToolTip("Open the Post-hoc Analysis Wizard for statistical analysis")
    button.clicked.connect(lambda: launch_analysis_wizard(parent))
    return button


# Example integration code for main application
"""
To integrate the wizard into your main application, add this to your main window setup:

from pyAPisolation.gui.wizard_integration import add_analysis_wizard_to_menu

class MainApplication(QMainWindow):
    def setupUi(self):
        # ... existing UI setup code ...
        
        # Add analysis wizard to Tools menu
        add_analysis_wizard_to_menu(self, self.menuTools)
        
        # ... rest of setup code ...
"""
