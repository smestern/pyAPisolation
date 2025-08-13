# %%
# This script applies EPSC thresholding to the time series data
import numpy as np
import scipy.stats as stats
import pyabf
import matplotlib.pyplot as plt

# %%
def epsc_threshold(data, thres=-5, rearm=None, min_length=0):
    """
    detect ESPCs via thresholding

    return thresholded
    """
    #get points where data crosses the threshold
    thresholded = np.argwhere(data < thres).flatten()
    #count the first instances of each continous threshold
    if rearm is None:
        diff = np.diff(thresholded, prepend=0)
        first_instances = np.argwhere(diff > 1).flatten()
        first_instances = thresholded[first_instances]
        
    elif rearm is not None:
        diff = np.diff(thresholded, prepend=0)
        first_instances = np.argwhere(diff > 1).flatten()
        first_instances = thresholded[first_instances]
        first_instances_adj = []
        last_point = -1
        for i in range(len(first_instances)):
            point = first_instances[i]
            if point < last_point:
                continue
            #get the next point in the data where the data dips below rearm
            next_point = np.argwhere(data[point:] > rearm).flatten()
            if len(next_point) == 0:
                first_instances_adj.append(len(data) - 1)
                continue
            else:
                next_point = next_point[0] + point
            #is it min number of samples apart?
            if next_point - last_point > min_length:
                last_point = next_point
                first_instances_adj.append(point)
            #skip points that are too close together
            if point - last_point < min_length:
                continue
        first_instances = np.array(first_instances_adj)
    else:
        first_instances = np.argwhere(diff > 1).flatten()
        first_instances = thresholded[first_instances]

    return thresholded, first_instances

# %%
FILE_PATH = "Z:\\Molsrv\\Julia\\Data\\Opto\\Opto Perifornical for Grant_2025\\ALL FILES\\2025_08_06_0023.abf"
baseline = (1.0, 1.01)
stim_time = (1.047, 1.057)

def plot_file(abf, first_instances):
    fig, ax = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
    factor_ = 30
    for x in abf.sweepList:
        abf.setSweep(x)
        first_inst_temp = first_instances[x]
        data = abf.sweepY
            #baseline the data by takeing the slope of the sweep
        baseline_slope = np.polyfit(abf.sweepX, data, 1)
        baseline_intercept = baseline_slope[1]
        data = data - (baseline_slope[0] * abf.sweepX + baseline_intercept)

        if factor_ is not None:
            factor = x*factor_
        elif factor_ is None:
            factor = 0.0
            print(factor_)

        ax[0].plot(abf.sweepX, data+factor, label='Data', c='k')
        ax[0].plot(abf.sweepX[first_inst_temp], data[first_inst_temp]+factor, 'ro', label='Thresholded')
        ax[0].axhline(y=-5+factor, color='r', linestyle='--', label='Threshold')
        ax[0].axhline(y=-2+factor, color='g', linestyle='--', label='Rearm')
        ax[0].set_xlim(baseline[0] - 0.01, baseline[1] + 0.01)
        ax[1].plot(abf.sweepX, data+factor, label='Data', c='k')
        ax[1].plot(abf.sweepX[first_inst_temp], data[first_inst_temp]+factor, 'ro', label='Thresholded')
        ax[1].axhline(y=-5+factor, color='r', linestyle='--', label='Threshold')
        ax[1].axhline(y=-2+factor, color='g', linestyle='--', label='Rearm')
        #plt.legend()
        ax[1].set_xlim(stim_time[0] - 0.01, stim_time[1]+ 0.01)
    ax[1].axvline(x=stim_time[0], color='b', linestyle='--', label='Stim Start')
    ax[1].axvline(x=stim_time[1], color='b', linestyle='--', label='Stim End')
    ax[1].set_ylim(-100, (factor)+10)

    plt.show()
    return first_instances

class InteractiveEPSCPlotter:
    def __init__(self, abf, first_instances, baseline=(1.0, 1.01), stim_time=(1.047, 1.057)):
        self.abf = abf
        self.original_first_instances = [fi.copy() if fi is not None else None for fi in first_instances]
        self.first_instances = [fi.copy() if fi is not None else None for fi in first_instances]
        self.baseline = baseline
        self.stim_time = stim_time
        self.factor_ = 30
        self.deleted_points = {i: [] for i in range(len(first_instances))}
        
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        self.fig.suptitle('Interactive EPSC Analysis - Click on red dots to delete them')
        
        # Store plot elements for updating
        self.data_lines = []
        self.threshold_points = []
        self.threshold_lines = []
        self.rearm_lines = []
        
        self.setup_plot()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Add control buttons
        self.add_controls()
        
    def setup_plot(self):
        """Initialize the plot with all sweeps"""
        self.ax[0].clear()
        self.ax[1].clear()
        
        for x in self.abf.sweepList:
            self.abf.setSweep(x)
            first_inst_temp = self.first_instances[x]
            data = self.abf.sweepY
            
            # Baseline the data by taking the slope of the sweep
            baseline_slope = np.polyfit(self.abf.sweepX, data, 1)
            baseline_intercept = baseline_slope[1]
            data = data - (baseline_slope[0] * self.abf.sweepX + baseline_intercept)
            
            factor = x * self.factor_
            
            # Plot data
            line1, = self.ax[0].plot(self.abf.sweepX, data + factor, label=f'Sweep {x}', c='k', alpha=0.7)
            line2, = self.ax[1].plot(self.abf.sweepX, data + factor, label=f'Sweep {x}', c='k', alpha=0.7)
            
            # Plot threshold crossings if they exist
            if first_inst_temp is not None and len(first_inst_temp) > 0:
                points1, = self.ax[0].plot(self.abf.sweepX[first_inst_temp], data[first_inst_temp] + factor, 
                                         'ro', markersize=8, picker=True, pickradius=10, label='Detections')
                points2, = self.ax[1].plot(self.abf.sweepX[first_inst_temp], data[first_inst_temp] + factor, 
                                         'ro', markersize=8, picker=True, pickradius=10, label='Detections')
                # Store sweep index in the artist for identification
                points1.sweep_idx = x
                points2.sweep_idx = x
                self.threshold_points.extend([points1, points2])
            
            # Plot threshold and rearm lines
            thresh_line1 = self.ax[0].axhline(y=-7+factor, color='r', linestyle='--', alpha=0.5, label='Threshold')
            thresh_line2 = self.ax[1].axhline(y=-7+factor, color='r', linestyle='--', alpha=0.5, label='Threshold')
            rearm_line1 = self.ax[0].axhline(y=-2+factor, color='g', linestyle='--', alpha=0.5, label='Rearm')
            rearm_line2 = self.ax[1].axhline(y=-2+factor, color='g', linestyle='--', alpha=0.5, label='Rearm')
        
        # Set plot limits and labels
        self.ax[0].set_xlim(self.baseline[0] - 0.01, self.baseline[1] + 0.01)
        self.ax[0].set_title('Baseline Period')
        self.ax[0].set_xlabel('Time (s)')
        self.ax[0].set_ylabel('Current (pA)')
        
        self.ax[1].set_xlim(self.stim_time[0] - 0.01, self.stim_time[1] + 0.01)
        self.ax[1].set_title('Stimulation Period')
        self.ax[1].set_xlabel('Time (s)')
        self.ax[1].axvline(x=self.stim_time[0], color='b', linestyle='--', alpha=0.7, label='Stim Start')
        self.ax[1].axvline(x=self.stim_time[1], color='b', linestyle='--', alpha=0.7, label='Stim End')
        
        max_sweep = len(self.abf.sweepList) - 1
        self.ax[1].set_ylim(-100, (max_sweep * self.factor_) + 10)
        
        plt.tight_layout()
        
    def add_controls(self):
        """Add control buttons to the plot"""
        from matplotlib.widgets import Button
        
        # Create button axes
        ax_reset = plt.axes([0.02, 0.02, 0.1, 0.04])
        ax_save = plt.axes([0.15, 0.02, 0.1, 0.04])
        ax_stats = plt.axes([0.28, 0.02, 0.1, 0.04])
        
        # Create buttons
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_save = Button(ax_save, 'Save')
        self.button_stats = Button(ax_stats, 'Stats')
        
        # Connect button events
        self.button_reset.on_clicked(self.reset_deletions)
        self.button_save.on_clicked(self.save_results)
        self.button_stats.on_clicked(self.show_stats)
        
    def on_click(self, event):
        """Handle click events on the plot"""
        if event.inaxes not in self.ax:
            return
            
        # Check if click was on a threshold point
        for artist in self.threshold_points:
            if artist.contains(event)[0]:
                sweep_idx = artist.sweep_idx
                
                # Find which point was clicked
                if self.first_instances[sweep_idx] is not None:
                    self.abf.setSweep(sweep_idx)
                    
                    # Get click coordinates
                    click_time = event.xdata
                    click_y = event.ydata
                    
                    # Find closest detection point
                    detection_times = self.abf.sweepX[self.first_instances[sweep_idx]]
                    if len(detection_times) > 0:
                        closest_idx = np.argmin(np.abs(detection_times - click_time))
                        point_to_delete = self.first_instances[sweep_idx][closest_idx]
                        
                        # Remove the point
                        self.deleted_points[sweep_idx].append(point_to_delete)
                        self.first_instances[sweep_idx] = np.delete(self.first_instances[sweep_idx], closest_idx)
                        
                        print(f"Deleted point at time {detection_times[closest_idx]:.4f}s in sweep {sweep_idx}")
                        
                        # Update the plot
                        self.update_plot()
                        break
    
    def update_plot(self):
        """Update the plot after deletion"""
        self.setup_plot()
        self.fig.canvas.draw()
        
    def reset_deletions(self, event):
        """Reset all deletions"""
        self.first_instances = [fi.copy() if fi is not None else None for fi in self.original_first_instances]
        self.deleted_points = {i: [] for i in range(len(self.first_instances))}
        print("Reset all deletions")
        self.update_plot()
        
    def save_results(self, event):
        """Save the current state"""
        print("Current first instances saved to self.first_instances")
        self.show_stats(None)
        
    def show_stats(self, event):
        """Show statistics about detections"""
        total_original = sum(len(fi) if fi is not None else 0 for fi in self.original_first_instances)
        total_current = sum(len(fi) if fi is not None else 0 for fi in self.first_instances)
        total_deleted = total_original - total_current
        
        print(f"\n=== Detection Statistics ===")
        print(f"Original detections: {total_original}")
        print(f"Current detections: {total_current}")
        print(f"Deleted detections: {total_deleted}")
        
        for sweep_idx in range(len(self.first_instances)):
            orig_count = len(self.original_first_instances[sweep_idx]) if self.original_first_instances[sweep_idx] is not None else 0
            curr_count = len(self.first_instances[sweep_idx]) if self.first_instances[sweep_idx] is not None else 0
            if orig_count > 0 or curr_count > 0:
                print(f"Sweep {sweep_idx}: {orig_count} -> {curr_count} ({orig_count - curr_count} deleted)")
        print("=" * 28)
        
    def get_results(self):
        """Return the current first instances"""
        return self.first_instances

def plot_file_interactive(abf, first_instances, baseline=(1.0, 1.01), stim_time=(1.047, 1.057)):
    """Create an interactive plot for EPSC analysis"""
    plotter = InteractiveEPSCPlotter(abf, first_instances, baseline, stim_time)
    plt.show()
    return plotter.get_results()


def process_file(file_path, interactive=False):

    abf = pyabf.ABF(file_path)

    sweep_first = []
    return_dict = {'pre_event_count': [], 'stim_event_count': [], 'threshold_idx': []}
    
    for x in abf.sweepList:
        abf.setSweep(x)
        data = abf.sweepY

        # Baseline the data by taking the slope of the sweep
        baseline_slope = np.polyfit(abf.sweepX, data, 1)
        baseline_intercept = baseline_slope[1]
        data = data - (baseline_slope[0] * abf.sweepX + baseline_intercept)

        # Try EPSC thresholding
        thresholded, first_instances = epsc_threshold(data, thres=-7, rearm=-2, min_length=100)
        sweep_first.append(first_instances if first_instances.size > 0 else None)

    if interactive:
        # Use interactive plotting for manual curation
        print(f"Opening interactive plot for {file_path}")
        print("Instructions:")
        print("- Click on red dots to delete false positive detections")
        print("- Use 'Reset' button to restore all deletions")
        print("- Use 'Stats' button to see detection counts") 
        print("- Close the plot window when finished")
        
        sweep_first = plot_file_interactive(abf, sweep_first, baseline, stim_time)
    else:
        # Use regular plotting
        plot_file(abf, sweep_first)

    # Calculate event counts with the (possibly modified) first instances
    for x in abf.sweepList:
        abf.setSweep(x)
        first_instances = sweep_first[x]
        
        if first_instances is not None and len(first_instances) > 0:
            event_times_sweep = abf.sweepX[first_instances]
        else:
            event_times_sweep = np.array([])

        # Get num events within baseline and stim periods
        num_events_baseline = np.sum((event_times_sweep >= baseline[0]) & (event_times_sweep < baseline[1]))
        num_events_stim = np.sum((event_times_sweep >= stim_time[0]) & (event_times_sweep < stim_time[1]))
        
        return_dict['pre_event_count'].append(num_events_baseline)
        return_dict['stim_event_count'].append(num_events_stim)
        return_dict['threshold_idx'].append(first_instances)
    
    return return_dict


# Test function for interactive plotting
def test_interactive_plot(file_path=None):
    """Test the interactive EPSC plotter with a single file"""
    if file_path is None:
        file_path = FILE_PATH
    
    print(f"Testing interactive plot with: {file_path}")
    results = process_file(file_path, interactive=True)
    return results


# %%
import glob
import pandas as pd
import os

XLSX_files = "Z:\\Molsrv\\Julia\\Data\\Opto\\Opto Perifornical for Grant_2025\\FILENAMES_Opto Perifornical_for_Yehor.xlsx"
ABF_ROOT = "Z:\\Molsrv\\Julia\\Data\\Opto\\Opto Perifornical for Grant_2025\\ALL FILES\\"
#load the Excel file
xls = pd.read_excel(XLSX_files, dtype={"PP (20 ms)": str})
#drop rows where the column "Postsynaptic" is labelled as "PS"
xls = xls[xls['Postsynaptic'] != "PS"]

print(xls.head())
pp_files = xls['PP (20 ms)'].tolist()
# Process files (set interactive=True for manual curation)
INTERACTIVE_MODE = False  # Set to True for interactive processing

result_dict = {}
for file in pp_files:
    file_path = ABF_ROOT + str(file) + ".abf"
    if os.path.exists(file_path):
        print(f"Processing file: {file_path}")
        results = process_file(file_path, interactive=INTERACTIVE_MODE)
        if not INTERACTIVE_MODE:
            plt.savefig(ABF_ROOT+"plots/"+str(file)+".png")
        plt.close('all')
        result_dict[file] = results
    else:
        print(f"!!!! File not found: {file_path} !!!!")
    
    # Break after first file if in interactive mode for testing
    if INTERACTIVE_MODE:
        print("Interactive mode - processing only first file for testing")
        break


# %%



