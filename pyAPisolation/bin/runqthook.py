import os
import sys
from pathlib import Path

os.environ['QT_PLUGIN_PATH'] = str( Path( sys._MEIPASS ) / 'PySide2' )
#os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str( Path( sys._MEIPASS ) / './platforms/' )