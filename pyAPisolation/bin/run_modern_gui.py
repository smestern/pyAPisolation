from pyAPisolation.gui.modern_gui import ModernAnalysisGUI
from PySide6.QtWidgets import QApplication

if __name__ == "__main__":
    app = QApplication([])
    gui = ModernAnalysisGUI(app)
    gui.main_widget.show()
    app.exec()