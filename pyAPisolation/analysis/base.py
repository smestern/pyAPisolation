"""
Base classes for the analysis framework.

This module contains the abstract base class that all analysis modules must inherit from.
"""

from abc import ABC, abstractmethod


class AnalysisModule(ABC):
    """
    Abstract base class for analysis modules.
    Each analysis type should inherit from this class and implement the required methods.
    """
    
    def __init__(self, name: str, display_name: str = None):
        self.name = name
        self.display_name = display_name or name
        self.param_dict = {}
        
    @abstractmethod
    def get_ui_elements(self):
        """
        Return a dictionary mapping UI element names to their expected types.
        This helps the GUI know what controls to bind for this analysis.
        
        Returns:
            dict: {element_name: element_type, ...}
        """
        pass
    
    @abstractmethod
    def parse_ui_params(self, ui_elements):
        """
        Parse parameters from UI elements into the format needed for analysis.
        
        Args:
            ui_elements: Dictionary of UI elements
            
        Returns:
            dict: Parameter dictionary for this analysis
        """
        pass
    
    @abstractmethod
    def run_individual_analysis(self, abf, selected_sweeps, param_dict, 
                               popup=None, show_rejected=False):
        """
        Run analysis on a single file for preview/individual analysis.
        
        Args:
            abf: The ABF file object
            selected_sweeps: List of sweep numbers to analyze
            param_dict: Analysis parameters
            popup: Progress dialog (optional)
            show_rejected: Whether to show rejected spikes (optional)
            
        Returns:
            dict: Results dictionary with analysis data
        """
        pass
    
    @abstractmethod
    def run_batch_analysis(self, folder_path, param_dict, protocol_name):
        """
        Run analysis on a folder of files.
        
        Args:
            folder_path: Path to folder containing files
            param_dict: Analysis parameters
            protocol_name: Protocol filter
            
        Returns:
            tuple: (dataframes, summary_data) - format depends on analysis type
        """
        pass
    
    @abstractmethod
    def save_results(self, results, output_dir, output_tag, save_options=None):
        """
        Save analysis results to files.
        
        Args:
            results: Results from batch analysis
            output_dir: Directory to save to
            output_tag: Tag to append to filenames
            save_options: Dictionary of save options (optional)
        """
        pass
    
    def get_plot_data(self, results, sweep_number=None):
        """
        Extract data for plotting from analysis results.
        Optional method - implement if analysis has specific plotting needs.
        
        Args:
            results: Analysis results
            sweep_number: Specific sweep to plot (optional)
            
        Returns:
            dict: Plot data or None for default plotting
        """
        return None
    
    def __str__(self):
        return f"AnalysisModule(name='{self.name}', display_name='{self.display_name}')"
    
    def __repr__(self):
        return self.__str__()
