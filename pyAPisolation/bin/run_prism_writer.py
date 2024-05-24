from PySide2.QtWidgets import QApplication
from pyAPisolation.dev.prism_writer_gui import PrismWriterGUI

if __name__ == '__main__':
    app = QApplication([])
    ex = PrismWriterGUI()
    app.exec_()