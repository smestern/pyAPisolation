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
        self.current_sweep = 0
        self.deleted_points = {i: [] for i in range(len(first_instances))}
        
        self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 8), sharey=True)
        self.update_title()
        
        # Store plot elements for updating
        self.threshold_points = []
        
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Add control buttons
        self.add_controls()
        self.setup_plot()
        
    def update_title(self):
        """Update the plot title with current sweep information"""
        total_sweeps = len(self.abf.sweepList)
        orig_count = len(self.original_first_instances[self.current_sweep]) if self.original_first_instances[self.current_sweep] is not None else 0
        curr_count = len(self.first_instances[self.current_sweep]) if self.first_instances[self.current_sweep] is not None else 0
        
        title = f'Interactive EPSC Analysis - Sweep {self.current_sweep + 1}/{total_sweeps} | '
        title += f'Detections: {curr_count}/{orig_count} | '
        title += f'Click red dots to delete | Arrow keys: prev/next sweep'
        self.fig.suptitle(title)
        
    def setup_plot(self):
        """Initialize the plot with current sweep"""
        self.ax[0].clear()
        self.ax[1].clear()
        self.threshold_points = []
        
        # Get current sweep data
        self.abf.setSweep(self.current_sweep)
        first_inst_temp = self.first_instances[self.current_sweep]
        data = self.abf.sweepY
        
        # Baseline the data by taking the slope of the sweep
        baseline_slope = np.polyfit(self.abf.sweepX, data, 1)
        baseline_intercept = baseline_slope[1]
        data = data - (baseline_slope[0] * self.abf.sweepX + baseline_intercept)
        
        # Plot data
        self.ax[0].plot(self.abf.sweepX, data, 'k-', linewidth=1.5, label=f'Sweep {self.current_sweep + 1}')
        self.ax[1].plot(self.abf.sweepX, data, 'k-', linewidth=1.5, label=f'Sweep {self.current_sweep + 1}')
        
        # Plot threshold crossings if they exist
        if first_inst_temp is not None and len(first_inst_temp) > 0:
            points1, = self.ax[0].plot(self.abf.sweepX[first_inst_temp], data[first_inst_temp], 
                                     'ro', markersize=10, picker=True, pickradius=15, 
                                     label=f'Detections ({len(first_inst_temp)})', zorder=5)
            points2, = self.ax[1].plot(self.abf.sweepX[first_inst_temp], data[first_inst_temp], 
                                     'ro', markersize=10, picker=True, pickradius=15,
                                     label=f'Detections ({len(first_inst_temp)})', zorder=5)
            # Store sweep index in the artist for identification
            points1.sweep_idx = self.current_sweep
            points2.sweep_idx = self.current_sweep
            self.threshold_points.extend([points1, points2])
        
        # Plot threshold and rearm lines
        self.ax[0].axhline(y=-7, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Threshold (-7 pA)')
        self.ax[1].axhline(y=-7, color='r', linestyle='--', alpha=0.7, linewidth=2, label='Threshold (-7 pA)')
        self.ax[0].axhline(y=-2, color='g', linestyle='--', alpha=0.7, linewidth=2, label='Rearm (-2 pA)')
        self.ax[1].axhline(y=-2, color='g', linestyle='--', alpha=0.7, linewidth=2, label='Rearm (-2 pA)')
        
        # Set plot limits and labels
        self.ax[0].set_xlim(self.baseline[0] - 0.01, self.baseline[1] + 0.01)
        self.ax[0].set_title('Baseline Period', fontsize=12, fontweight='bold')
        self.ax[0].set_xlabel('Time (s)')
        self.ax[0].set_ylabel('Current (pA)')
        self.ax[0].grid(True, alpha=0.3)
        self.ax[0].legend(loc='upper right', fontsize=10)
        
        self.ax[1].set_xlim(self.stim_time[0] - 0.01, self.stim_time[1] + 0.01)
        self.ax[1].set_title('Stimulation Period', fontsize=12, fontweight='bold')
        self.ax[1].set_xlabel('Time (s)')
        self.ax[1].axvline(x=self.stim_time[0], color='b', linestyle='--', alpha=0.7, linewidth=2, label='Stim Start')
        self.ax[1].axvline(x=self.stim_time[1], color='b', linestyle='--', alpha=0.7, linewidth=2, label='Stim End')
        self.ax[1].grid(True, alpha=0.3)
        self.ax[1].legend(loc='upper right', fontsize=10)
        
        # Auto-scale y-axis with some padding
        y_min, y_max = np.min(data), np.max(data)
        y_range = y_max - y_min
        padding = y_range * 0.1
        self.ax[0].set_ylim(y_min - padding, y_max + padding)
        self.ax[1].set_ylim(y_min - padding, y_max + padding)
        
        plt.tight_layout()
        self.update_title()
        
    def add_controls(self):
        """Add control buttons to the plot"""
        from matplotlib.widgets import Button
        
        # Create button axes
        ax_prev = plt.axes([0.02, 0.02, 0.08, 0.04])
        ax_next = plt.axes([0.12, 0.02, 0.08, 0.04])
        ax_reset = plt.axes([0.22, 0.02, 0.08, 0.04])
        ax_reset_all = plt.axes([0.32, 0.02, 0.08, 0.04])
        ax_stats = plt.axes([0.42, 0.02, 0.08, 0.04])
        ax_save = plt.axes([0.52, 0.02, 0.08, 0.04])
        
        # Create buttons
        self.button_prev = Button(ax_prev, '← Prev')
        self.button_next = Button(ax_next, 'Next →')
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset_all = Button(ax_reset_all, 'Reset All')
        self.button_stats = Button(ax_stats, 'Stats')
        self.button_save = Button(ax_save, 'Save')
        
        # Connect button events
        self.button_prev.on_clicked(self.prev_sweep)
        self.button_next.on_clicked(self.next_sweep)
        self.button_reset.on_clicked(self.reset_current_sweep)
        self.button_reset_all.on_clicked(self.reset_all_deletions)
        self.button_stats.on_clicked(self.show_stats)
        self.button_save.on_clicked(self.save_results)
        
    def on_key_press(self, event):
        """Handle keyboard navigation"""
        if event.key == 'left':
            self.prev_sweep(None)
        elif event.key == 'right':
            self.next_sweep(None)
        elif event.key == 'r':
            self.reset_current_sweep(None)
        elif event.key == 's':
            self.show_stats(None)
            
    def prev_sweep(self, event):
        """Go to previous sweep"""
        if self.current_sweep > 0:
            self.current_sweep -= 1
            self.setup_plot()
            self.fig.canvas.draw()
        
    def next_sweep(self, event):
        """Go to next sweep"""
        if self.current_sweep < len(self.abf.sweepList) - 1:
            self.current_sweep += 1
            self.setup_plot()
            self.fig.canvas.draw()
        
    def on_click(self, event):
        """Handle click events on the plot"""
        if event.inaxes not in self.ax:
            return
            
        # Only process left clicks for deletion
        if event.button != 1:
            return

        # Only process clicks for the current sweep
        sweep_idx = self.current_sweep
        
        # Check if click was on a threshold point
        for artist in self.threshold_points:
            if artist.contains(event)[0] and artist.sweep_idx == sweep_idx:
                
                # Find which point was clicked
                if self.first_instances[sweep_idx] is not None:
                    self.abf.setSweep(sweep_idx)
                    
                    # Get click coordinates
                    click_time = event.xdata
                    
                    # Find closest detection point
                    detection_times = self.abf.sweepX[self.first_instances[sweep_idx]]
                    if len(detection_times) > 0:
                        closest_idx = np.argmin(np.abs(detection_times - click_time))
                        
                        # Only delete if click is reasonably close to a point
                        if np.abs(detection_times[closest_idx] - click_time) < 0.01:
                            point_to_delete = self.first_instances[sweep_idx][closest_idx]
                            
                            # Remove the point
                            self.deleted_points[sweep_idx].append(point_to_delete)
                            self.first_instances[sweep_idx] = np.delete(
                                self.first_instances[sweep_idx], closest_idx)
                            
                            print(f"Deleted point at time {detection_times[closest_idx]:.4f}s "
                                  f"in sweep {sweep_idx + 1}")
                            
                            # Update the plot
                            self.setup_plot()
                            self.fig.canvas.draw()
                            break
    
        
    def update_plot(self):
        """Update the plot after changes"""
        self.setup_plot()
        self.fig.canvas.draw()
        
    def reset_current_sweep(self, event):
        """Reset deletions for current sweep only"""
        sweep_idx = self.current_sweep
        if self.original_first_instances[sweep_idx] is not None:
            self.first_instances[sweep_idx] = self.original_first_instances[sweep_idx].copy()
            self.deleted_points[sweep_idx] = []
            print(f"Reset deletions for sweep {sweep_idx + 1}")
            self.setup_plot()
            self.fig.canvas.draw()
        
    def reset_all_deletions(self, event):
        """Reset all deletions across all sweeps"""
        self.first_instances = [fi.copy() if fi is not None else None 
                               for fi in self.original_first_instances]
        self.deleted_points = {i: [] for i in range(len(self.first_instances))}
        print("Reset all deletions across all sweeps")
        self.setup_plot()
        self.fig.canvas.draw()
        
    def save_results(self, event):
        """Save the current state and show summary"""
        print("\n=== Results Saved ===")
        self.show_stats(None)
        
    def show_stats(self, event):
        """Show statistics about detections"""
        total_original = sum(len(fi) if fi is not None else 0 for fi in self.original_first_instances)
        total_current = sum(len(fi) if fi is not None else 0 for fi in self.first_instances)
        total_deleted = total_original - total_current
        
        print(f"\n=== Detection Statistics ===")
        print(f"Total original detections: {total_original}")
        print(f"Total current detections: {total_current}")
        print(f"Total deleted detections: {total_deleted}")
        print(f"\nCurrent sweep: {self.current_sweep + 1}/{len(self.abf.sweepList)}")
        
        # Show current sweep stats
        orig_count = len(self.original_first_instances[self.current_sweep]) if self.original_first_instances[self.current_sweep] is not None else 0
        curr_count = len(self.first_instances[self.current_sweep]) if self.first_instances[self.current_sweep] is not None else 0
        print(f"Current sweep detections: {curr_count}/{orig_count} ({orig_count - curr_count} deleted)")
        
        # Show summary for all sweeps with changes
        changed_sweeps = []
        for sweep_idx in range(len(self.first_instances)):
            orig_count = len(self.original_first_instances[sweep_idx]) if self.original_first_instances[sweep_idx] is not None else 0
            curr_count = len(self.first_instances[sweep_idx]) if self.first_instances[sweep_idx] is not None else 0
            if orig_count != curr_count:
                changed_sweeps.append((sweep_idx, orig_count, curr_count))
        
        if changed_sweeps:
            print(f"\nSweeps with changes:")
            for sweep_idx, orig, curr in changed_sweeps:
                print(f"  Sweep {sweep_idx + 1}: {orig} -> {curr} ({orig - curr} deleted)")
        else:
            print("\nNo changes made yet.")
        print("=" * 35)
        
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
        print("- Use arrow keys or buttons to navigate between sweeps")
        print("- Click on red dots to delete false positive detections")
        print("- Use 'Reset' button to restore deletions for current sweep")
        print("- Use 'Reset All' button to restore all deletions")
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



XLSX_files = "Z:\\Molsrv\\Julia\\Data\\Opto\\Opto Perifornical for Grant_2025\\FILENAMES_Opto Perifornical_for_Yehor.xlsx"
ABF_ROOT = "Z:\\Molsrv\\Julia\\Data\\Opto\\Opto Perifornical for Grant_2025\\ALL FILES\\"
#load the Excel file
xls = pd.read_excel(XLSX_files, dtype={"PP (20 ms)": str})
#drop rows where the column "Postsynaptic" is labelled as "PS"
xls = xls[xls['Postsynaptic'] != "PS"]

print(xls.head())
pp_files = xls['PP (20 ms)'].tolist()
# Process files (set interactive=True for manual curation)
INTERACTIVE_MODE = True # Set to True for interactive processing

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
    



# %%



