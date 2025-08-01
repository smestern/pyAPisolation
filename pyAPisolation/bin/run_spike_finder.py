
import multiprocessing

def main():
    # This function is a placeholder for the main functionality of the spike finder.
    # It can be replaced with the actual implementation.
    print("Spike Finder GUI is starting...")
    from pyAPisolation.gui.spikeFinder import main as spike_finder_main
    spike_finder_main()

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Ensure compatibility with frozen applications
    
    main()
# This script serves as an entry point to run the spike finder GUI.
# It imports the main function from the spikeFinder module and executes it.
