"""
Base classes for the analysis framework

This module defines the core interfaces and data structures that all
analyzers must implement.
"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np


@dataclass
class Parameter:
    """Individual parameter definition with metadata"""
    
    name: str
    param_type: type = None
    default: Any = None
    value: Any = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    description: Optional[str] = None
    options: Optional[List[Any]] = None  # For choice/enum parameters
    
    def __post_init__(self):
        """Validate parameter definition"""
        if self.min_value is not None and self.max_value is not None:
            if self.min_value > self.max_value:
                msg = (f"min_value ({self.min_value}) cannot be greater "
                       f"than max_value ({self.max_value})")
                raise ValueError(msg)
        # Validate default value
        if self.default is not None:
            if not self.is_valid(self.default):
                msg = (f"Default value {self.default} is not valid "
                    f"for parameter {self.name}")
                raise ValueError(msg)
        
        if self.value is None and self.default is not None:
            self.value = self.default
    
    def is_valid(self, value: Any) -> bool:
        """Check if a value is valid for this parameter"""
        # Type check
        if not isinstance(value, self.param_type):
            try:
                # Try to convert to the expected type
                value = self.param_type(value)
            except (ValueError, TypeError):
                return False
        
        # Range check for numeric types
        if (self.min_value is not None and hasattr(value, '__lt__')
                and value < self.min_value):
            return False
        if (self.max_value is not None and hasattr(value, '__gt__')
                and value > self.max_value):
            return False
        
        # Options check
        if self.options is not None and value not in self.options:
            return False
        
        return True
    
    def validate_and_convert(self, value: Any) -> Any:
        """Validate and convert a value to the correct type"""
        if not isinstance(value, self.param_type):
            try:
                value = self.param_type(value)
            except (ValueError, TypeError):
                msg = (f"Cannot convert {value} to type "
                       f"{self.param_type.__name__} for parameter {self.name}")
                raise ValueError(msg)
        
        if not self.is_valid(value):
            constraints = []
            if self.min_value is not None:
                constraints.append(f"min: {self.min_value}")
            if self.max_value is not None:
                constraints.append(f"max: {self.max_value}")
            if self.options is not None:
                constraints.append(f"options: {self.options}")
            
            constraint_str = ", ".join(constraints)
            msg = (f"Value {value} is not valid for parameter {self.name}. "
                   f"Constraints: {constraint_str}")
            raise ValueError(msg)
        
        return value
    
    def get(self, key, val=None):
        if hasattr(self, key):
            return getattr(self, key)
        else:
            return val
        
    def set(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            #create the property?
            self.__dict__[key] = value


@dataclass
class AnalysisParameters:
    """Parameters for analysis configuration"""
    
    # Common parameters
    start_time: Parameter = Parameter("start_time", param_type=float, default=0.0, min_value=0.0, max_value=np.inf, description="Time in (s) to start analysis")
    end_time: Parameter = Parameter("end_time", param_type=float, default=0.0, min_value=0.0, max_value=np.inf, description="Time in (s) to end analysis")
    protocol_filter: Parameter = Parameter("protocol_filter", param_type=str, default="", description="Protocol filter")

    # Additional parameters stored as dict for flexibility
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def list(self) -> List[str]:
        """List all parameter names including extra params"""
        common_parameters = [x for x in self.__dataclass_fields__.keys() if x != "extra_params"]
        return common_parameters + list(self.extra_params.keys())

    def get(self, key: str, default: Any = None) -> Any:
        """Get parameter value with fallback to extra_params"""
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra_params.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set parameter value"""
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            #create a new parameter instance
            temp_parameter = Parameter(name=key, param_type=type(value), value=value)
            self.extra_params[key] = temp_parameter


@dataclass
class AnalysisResult:
    """Container for analysis results"""
    
    analyzer_name: str
    file_path: str
    success: bool
    
    # Main results
    summary_data: Optional[pd.DataFrame] = None
    detailed_data: Optional[pd.DataFrame] = None
    sweep_data: Optional[pd.DataFrame] = None
    
    # Metadata and diagnostics
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def add_error(self, error: str) -> None:
        """Add an error message"""
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message"""
        self.warnings.append(warning)


class AnalysisModule:
    """
    Abstract base class for analysis modules.
    Each analysis type should inherit from this class and implement the
    required methods.
    """
    
    def __init__(self, name: str, display_name: str = None,
                 parameters: Optional[AnalysisParameters] = None):
        self.name = name
        self.display_name = display_name or name
        self._parameters = parameters or AnalysisParameters()
        # Keep param_dict for backward compatibility
        self.param_dict = {}
        
    @property
    def parameters(self) -> AnalysisParameters:
        """Get the analysis parameters object"""
        return self._parameters
    
    @parameters.setter
    def parameters(self, value: AnalysisParameters) -> None:
        """Set the analysis parameters object"""
        if not isinstance(value, AnalysisParameters):
            raise TypeError(
                "Parameters must be an AnalysisParameters instance")
        self._parameters = value
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """
        Get a specific parameter value
        
        Args:
            key: Parameter name
            default: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        return self._parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        """
        Set a specific parameter value
        
        Args:
            key: Parameter name
            value: Parameter value
        """
        self._parameters.set(key, value)
    
    def update_parameters(self, **kwargs) -> None:
        """
        Update multiple parameters at once
        
        Args:
            **kwargs: Parameter key-value pairs
        """
        for key, value in kwargs.items():
            self.set_parameter(key, value)
    
    def reset_parameters(self) -> None:
        """Reset parameters to default values"""
        self._parameters = AnalysisParameters()

    def get_ui_elements(self):
        """
        Return a dictionary mapping UI element names to their expected types
        and default values. This helps the GUI know what controls to bind
        for this analysis.
        
        Returns:
            dict: {element_name: element_type, ...}
        """
        ui_params = {}
        for param_name in self._parameters.list():
            # param_type = self._parameters.get(param_name).param_type
            # param_value = self._parameters.get(param_name).value
            ui_params[param_name] = self._parameters.get(param_name)
        return ui_params
    
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
        return (f"AnalysisModule(name='{self.name}', "
                f"display_name='{self.display_name}')")
    
    def __repr__(self):
        return self.__str__()

