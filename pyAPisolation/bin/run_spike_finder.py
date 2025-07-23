import matplotlib
#from  .mainwindow import Ui_MainWindow
matplotlib.use('QtAgg')
from pyAPisolation.gui.spikeFinder import main

if __name__ == "__main__":
    main()
# This script serves as an entry point to run the spike finder GUI.
# It imports the main function from the spikeFinder module and executes it.
