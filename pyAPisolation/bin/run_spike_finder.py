
import multiprocessing


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Ensure compatibility with frozen applications
    from pyAPisolation.gui.spikeFinder import main
    main()
# This script serves as an entry point to run the spike finder GUI.
# It imports the main function from the spikeFinder module and executes it.
