#!/usr/bin/env python3
"""
Example script showing how to add a custom analysis module to the pyAPisolation GUI.

This script demonstrates:
1. Creating a custom analysis module
2. Registering it with the analysis registry
3. Running the GUI with the new analysis available

Usage:
    python add_custom_analysis.py
"""

import sys
import os

# Add the pyAPisolation package to the path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyAPisolation.analysis import AnalysisModule, analysis_registry
import numpy as np
import pandas as pd


class SimpleVoltageStatsModule(AnalysisModule):
    """
    Simple example: Calculate basic voltage statistics
    """
    
    def __init__(self):
        super().__init__("voltage_stats", "Voltage Statistics")
    
    def get_ui_elements(self):
        return {
            'start_time': 'float',
            'end_time': 'float',
            'bessel': 'float'
        }
    
    def parse_ui_params(self, ui_elements):
        start = float(ui_elements.get('start_time', {}).text() or "0")
        end = float(ui_elements.get('end_time', {}).text() or "1000") 
        bessel = float(ui_elements.get('bessel', {}).text() or "0")
        
        return {
            'start': start / 1000,  # Convert ms to s
            'end': end / 1000,      # Convert ms to s
            'bessel_filter': bessel
        }
    
    def run_individual_analysis(self, abf, selected_sweeps, param_dict, 
                               popup=None, show_rejected=False):
        if selected_sweeps is None:
            selected_sweeps = abf.sweepList
        
        results = []
        
        for i, sweep in enumerate(selected_sweeps):
            abf.setSweep(sweep)
            
            # Find time indices
            start_idx = np.searchsorted(abf.sweepX, param_dict['start'])
            end_idx = np.searchsorted(abf.sweepX, param_dict['end'])
            
            # Extract voltage in the analysis window
            voltage = abf.sweepY[start_idx:end_idx]
            
            # Calculate statistics
            stats = {
                'sweep': sweep,
                'mean_voltage': np.mean(voltage),
                'std_voltage': np.std(voltage),
                'min_voltage': np.min(voltage),
                'max_voltage': np.max(voltage),
                'rms_voltage': np.sqrt(np.mean(voltage**2)),
                'range_voltage': np.max(voltage) - np.min(voltage)
            }
            
            results.append(stats)
            
            if popup:
                popup.setValue(i + 1)
        
        return {
            'voltage_stats': pd.DataFrame(results),
            'spike_df': None,
            'rejected_spikes': None,
            'subthres_df': None
        }
    
    def run_batch_analysis(self, folder_path, param_dict, protocol_name):
        """
        For now, raise NotImplementedError to use legacy fallback
        In a full implementation, you'd process all files in the folder
        """
        raise NotImplementedError("Batch analysis not yet implemented for voltage stats")
    
    def save_results(self, results, output_dir, output_tag, save_options=None):
        import os
        
        if isinstance(results, pd.DataFrame) and not results.empty:
            output_path = os.path.join(output_dir, f"voltage_stats_{output_tag}.csv")
            results.to_csv(output_path, index=False)
            print(f"Voltage statistics saved to: {output_path}")


def register_custom_analysis():
    """Register the custom voltage statistics analysis using the utility functions"""
    print("Registering custom voltage statistics analysis...")
    
    # Method 1: Using the utility function
    from pyAPisolation.analysis import register_analysis_module
    voltage_module = register_analysis_module(SimpleVoltageStatsModule)
    
    # Method 2: Manual registration (alternative approach)
    # from pyAPisolation.analysis import analysis_registry
    # voltage_module = SimpleVoltageStatsModule()
    # analysis_registry.register_module(voltage_module)
    
    # Method 3: Register with tab mapping (if you want to add it to a specific tab)
    # from pyAPisolation.analysis import register_analysis_with_tab
    # voltage_module = register_analysis_with_tab(SimpleVoltageStatsModule, 2)  # Tab index 2
    
    print(f"Successfully registered: {voltage_module.display_name}")
    
    # Show all available modules
    from pyAPisolation.analysis import list_available_analyses
    print(f"Available modules: {list_available_analyses()}")


def register_using_decorator():
    """Example of using the decorator approach for registration"""
    from pyAPisolation.analysis import analysis_module, AnalysisModule
    import numpy as np
    import pandas as pd
    
    @analysis_module(name="decorator_stats", display_name="Decorator Statistics")
    class DecoratorStatsModule(AnalysisModule):
        """Example module registered using decorator"""
        
        def get_ui_elements(self):
            return {
                'start_time': 'float',
                'end_time': 'float'
            }
        
        def parse_ui_params(self, ui_elements):
            return {
                'start': float(ui_elements.get('start_time', {}).text() or "0") / 1000,
                'end': float(ui_elements.get('end_time', {}).text() or "1000") / 1000
            }
        
        def run_individual_analysis(self, abf, selected_sweeps, param_dict, 
                                   popup=None, show_rejected=False):
            return {'decorator_results': pd.DataFrame()}
        
        def run_batch_analysis(self, folder_path, param_dict, protocol_name):
            raise NotImplementedError("Decorator example - batch analysis not implemented")
        
        def save_results(self, results, output_dir, output_tag, save_options=None):
            print(f"Decorator module saving results to {output_dir}")
    
    print("Decorator-registered module is now available!")
    return DecoratorStatsModule


def main():
    """Main function to demonstrate adding custom analysis with new utilities"""
    print("=== Custom Analysis Module Demo with Registration Utilities ===\n")
    
    # Method 1: Register using utility function
    print("1. Registering with utility function...")
    register_custom_analysis()
    
    # Method 2: Register using decorator
    print("\n2. Registering with decorator...")
    register_using_decorator()
    
    print("\n=== Demonstrating Registry Features ===")
    
    # Show all available modules
    from pyAPisolation.analysis import list_available_analyses, get_analysis_module
    print(f"All available modules: {list_available_analyses()}")
    
    # Get a specific module
    voltage_module = get_analysis_module('voltage_stats')
    if voltage_module:
        print(f"Retrieved module: {voltage_module.display_name}")
        print(f"Required UI elements: {voltage_module.get_ui_elements()}")
    
    # Test the utility functions
    print("\n=== Testing Utility Functions ===")
    
    # Example of creating mock UI elements for testing
    class MockUIElement:
        def __init__(self, value):
            self._value = str(value)
        def text(self):
            return self._value
    
    mock_ui = {
        'start_time': MockUIElement(100),  # 100 ms
        'end_time': MockUIElement(500),    # 500 ms  
        'bessel': MockUIElement(0)         # No filter
    }
    
    if voltage_module:
        params = voltage_module.parse_ui_params(mock_ui)
        print(f"Parsed parameters: {params}")
    
    print("\n=== Integration Information ===")
    print("The new registration utilities provide several ways to add modules:")
    print("1. register_analysis_module(ModuleClass) - Simple registration")
    print("2. register_analysis_with_tab(ModuleClass, tab_index) - Register with tab")
    print("3. @analysis_module decorator - Automatic registration")
    print("4. Manual registration via analysis_registry.register_module()")
    
    print("\nTo run the GUI with your custom analyses:")
    print("from pyAPisolation.gui.spikeFinder import main")
    print("main()")
    
    # Uncomment to actually start the GUI:
    # print("\nStarting GUI with custom analyses...")
    # from pyAPisolation.gui.spikeFinder import main
    # main()


if __name__ == "__main__":
    main()
