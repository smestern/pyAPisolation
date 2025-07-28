"""
Example custom analysis module for the pyAPisolation GUI
This demonstrates how to create a new analysis type using the modular framework
"""

import numpy as np
import pandas as pd
from pyAPisolation.analysis import AnalysisModule, analysis_registry


class ExampleCustomAnalysisModule(AnalysisModule):
    """
    Example custom analysis module that demonstrates the framework.
    This module calculates basic statistics (mean, std, min, max) of voltage traces.
    """
    
    def __init__(self):
        super().__init__("custom_stats", "Custom Statistics Analysis")
        
    def get_ui_elements(self):
        """Define what UI elements this analysis needs"""
        return {
            'start_time': 'float',      # Start time for analysis window
            'end_time': 'float',        # End time for analysis window
            'bessel': 'float',          # Bessel filter frequency
            'custom_threshold': 'float'  # Custom threshold parameter
        }
    
    def parse_ui_params(self, ui_elements):
        """Parse UI elements into analysis parameters"""
        start_time = float(ui_elements.get('start_time', {}).text() or "0")
        end_time = float(ui_elements.get('end_time', {}).text() or "1000")
        bessel_filter = float(ui_elements.get('bessel', {}).text() or "0")
        threshold = float(ui_elements.get('custom_threshold', {}).text() or "0")
        
        return {
            'start': start_time,
            'end': end_time,
            'bessel_filter': bessel_filter,
            'threshold': threshold
        }
    
    def run_individual_analysis(self, abf, selected_sweeps, param_dict, 
                               popup=None, show_rejected=False):
        """Run analysis on individual file"""
        if selected_sweeps is None:
            selected_sweeps = abf.sweepList
            
        results = {}
        
        for sweep_idx, sweep in enumerate(selected_sweeps):
            abf.setSweep(sweep)
            
            # Get time window indices
            start_idx = np.searchsorted(abf.sweepX, param_dict['start'])
            end_idx = np.searchsorted(abf.sweepX, param_dict['end'])
            
            # Extract data window
            voltage_window = abf.sweepY[start_idx:end_idx]
            current_window = abf.sweepC[start_idx:end_idx]
            
            # Calculate basic statistics
            stats = {
                'sweep': sweep,
                'v_mean': np.mean(voltage_window),
                'v_std': np.std(voltage_window),
                'v_min': np.min(voltage_window),
                'v_max': np.max(voltage_window),
                'c_mean': np.mean(current_window),
                'c_std': np.std(current_window),
                'threshold_crossings': np.sum(
                    voltage_window > param_dict['threshold']
                )
            }
            
            results[sweep] = stats
            
            if popup:
                popup.setValue(sweep_idx)
        
        return {
            'stats_df': pd.DataFrame.from_dict(results, orient='index'),
            'spike_df': None,
            'rejected_spikes': None,
            'subthres_df': None
        }
    
    def run_batch_analysis(self, folder_path, param_dict, protocol_name):
        """Run analysis on folder of files"""
        import glob
        import os
        from pyAPisolation.dataset import cellData
        
        # Find all ABF files
        abf_files = glob.glob(os.path.join(folder_path, "**/*.abf"), 
                             recursive=True)
        
        all_results = []
        
        for file_path in abf_files:
            try:
                # Load file
                cell_data = cellData(file=file_path)
                
                # Filter by protocol if specified
                if protocol_name and protocol_name != "[No Filter]":
                    if protocol_name not in cell_data.protocol:
                        continue
                
                # Run individual analysis
                results = self.run_individual_analysis(
                    cell_data, None, param_dict
                )
                
                # Add file information
                stats_df = results['stats_df']
                stats_df['filename'] = os.path.basename(file_path)
                stats_df['filepath'] = file_path
                
                all_results.append(stats_df)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if all_results:
            combined_df = pd.concat(all_results, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def save_results(self, results, output_dir, output_tag, save_options=None):
        """Save analysis results"""
        import os
        
        if isinstance(results, pd.DataFrame) and not results.empty:
            output_path = os.path.join(
                output_dir, 
                f"custom_stats_analysis_{output_tag}.csv"
            )
            results.to_csv(output_path, index=False)
            print(f"Custom statistics analysis saved to: {output_path}")
    
    def get_plot_data(self, results, sweep_number=None):
        """Extract plot data for visualization"""
        if 'stats_df' in results:
            stats_df = results['stats_df']
            if sweep_number is not None and sweep_number in stats_df.index:
                return stats_df.loc[sweep_number].to_dict()
            return stats_df.to_dict('records')
        return None


# Example of how to register the new module
def register_custom_analysis():
    """Register the custom analysis module"""
    custom_module = ExampleCustomAnalysisModule()
    analysis_registry.register_module(custom_module)
    
    # If you want to add it to a specific tab, you can do:
    # analysis_registry.add_tab_mapping(2, "custom_stats")  # Tab index 2
    
    print(f"Registered custom analysis module: {custom_module.display_name}")


# Example of creating a more complex analysis module
class AdvancedCustomAnalysisModule(AnalysisModule):
    """
    More advanced example showing how to implement complex analysis
    """
    
    def __init__(self):
        super().__init__("advanced_custom", "Advanced Custom Analysis")
        
    def get_ui_elements(self):
        return {
            'start_time': 'float',
            'end_time': 'float',
            'analysis_method': 'combo',  # ComboBox for method selection
            'parameter1': 'float',
            'parameter2': 'float',
            'enable_filtering': 'bool'
        }
    
    def parse_ui_params(self, ui_elements):
        params = {}
        
        # Handle different UI element types
        for name, element in ui_elements.items():
            if hasattr(element, 'text'):  # LineEdit, etc.
                try:
                    params[name] = float(element.text())
                except (ValueError, AttributeError):
                    params[name] = element.text()
            elif hasattr(element, 'isChecked'):  # CheckBox
                params[name] = element.isChecked()
            elif hasattr(element, 'currentText'):  # ComboBox
                params[name] = element.currentText()
            elif hasattr(element, 'value'):  # SpinBox, etc.
                params[name] = element.value()
        
        return params
    
    def run_individual_analysis(self, abf, selected_sweeps, param_dict, 
                               popup=None, show_rejected=False):
        """Implement your custom analysis logic here"""
        # This is where you'd implement your specific analysis
        # For now, return a placeholder
        return {
            'custom_results': pd.DataFrame(),
            'spike_df': None,
            'rejected_spikes': None,
            'subthres_df': None
        }
    
    def run_batch_analysis(self, folder_path, param_dict, protocol_name):
        """Implement batch analysis"""
        # Implement your batch processing logic
        return pd.DataFrame()
    
    def save_results(self, results, output_dir, output_tag, save_options=None):
        """Implement custom save logic"""
        pass
