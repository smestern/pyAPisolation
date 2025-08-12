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
import glob
import logging
import pyabf
from ..dataset import cellData
import copy
logger = logging.getLogger(__name__)


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

    # --- Dict-like behavior ---
    def __getitem__(self, key):
        if hasattr(self, key):
            return getattr(self, key)
        elif key in self.extra_params:
            return self.extra_params[key]
        else:
            raise KeyError(key)

    def __setitem__(self, key, value):
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            temp_parameter = Parameter(name=key, param_type=type(value), value=value)
            self.extra_params[key] = temp_parameter

    def __contains__(self, key):
        return hasattr(self, key) or key in self.extra_params

    def __iter__(self):
        for k in self.__dataclass_fields__:
            if k != "extra_params":
                yield k
        for k in self.extra_params:
            yield k

    def keys(self):
        return list(iter(self))

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]


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

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary for serialization"""
        return {
            'analyzer_name': self.analyzer_name,
            'file_path': self.file_path,
            'success': self.success,
            'summary_data': self.summary_data.to_dict() if self.summary_data is not None else None,
            'detailed_data': self.detailed_data.to_dict() if self.detailed_data is not None else None,
            'sweep_data': self.sweep_data.to_dict() if self.sweep_data is not None else None,
            'metadata': self.metadata,
            'errors': self.errors,
            'warnings': self.warnings
        }

    def __add__(self, other: 'AnalysisResult') -> 'AnalysisResult':
        """Combine two AnalysisResults"""
        _detailed_df_list = []
        _summary_df_list = []
        _sweep_df_list = []
        _file_paths = set()
        _errors = []
        for res in [self, other]:
            if res.detailed_data is not None:
                _detailed_df_list.append(res.detailed_data)
            if res.summary_data is not None:
                _summary_df_list.append(res.summary_data)
            if res.sweep_data is not None:
                _sweep_df_list.append(res.sweep_data)
            if res.file_path:
                _file_paths.add(res.file_path)
            if res.errors:
                _errors.extend(res.errors)

        combined = AnalysisResult(
            analyzer_name=self.analyzer_name,
            file_path=_file_paths.pop() if _file_paths else None,
            success=all(res.success for res in [self, other]),
            summary_data=pd.concat(_summary_df_list) if _summary_df_list else None,
            detailed_data=pd.concat(_detailed_df_list) if _detailed_df_list else None,
            sweep_data=pd.concat(_sweep_df_list) if _sweep_df_list else None,
            metadata={**self.metadata, **other.metadata},
            errors=_errors,
            warnings=self.warnings + other.warnings
        )
        return combined

    @classmethod
    def concatenate(cls, results: List['AnalysisResult']) -> 'AnalysisResult':
        """Concatenate multiple AnalysisResults into one"""
        if not results:
            return cls(analyzer_name='NoAnalyzer', file_path='NoFile', success=False)
        
        _detailed_df_list = []
        _summary_df_list = []
        _sweep_df_list = []
        _file_paths = []
        _errors = []
        for res in results:
            if res.detailed_data is not None:
                _detailed_df_list.append(res.detailed_data)
            if res.summary_data is not None:
                _summary_df_list.append(res.summary_data)
            if res.sweep_data is not None:
                _sweep_df_list.append(res.sweep_data)
            if res.file_path:
                _file_paths.append(res.file_path)
            if res.errors:
                _errors.extend(res.errors)

        combined = AnalysisResult(
            analyzer_name=results[0].analyzer_name if results else 'NoAnalyzer',
            file_path=_file_paths,
            success=all(res.success for res in results),
            summary_data=pd.concat(_summary_df_list) if _summary_df_list else None,
            detailed_data=pd.concat(_detailed_df_list) if _detailed_df_list else None,
            sweep_data=pd.concat(_sweep_df_list) if _sweep_df_list else None,
            metadata={},
            errors=_errors,
            warnings=[warning for res in results for warning in res.warnings]
        )
        
        return combined


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
        
    def analyze(self, x=None, y=None, c=None, file=None, celldata=None, 
                selected_sweeps=None, **kwargs) -> AnalysisResult:
        """
        Unified analysis interface that accepts various input types.
        
        Args:
            x (np.array, optional): Time array of the sweep(s)
            y (np.array, optional): Voltage array of the sweep(s) 
            c (np.array, optional): Current array of the sweep(s)
            file (str, optional): File path to analyze
            celldata (cellData, optional): cellData object to analyze
            selected_sweeps (list, optional): List of sweep numbers to analyze
            **kwargs: Additional parameters to override defaults
            
        Returns:
            AnalysisResult: Container with analysis results and metadata
        """
        from ..patch_utils import parse_user_input  # Import here to avoid circular imports
        
        # Create result container
        result = AnalysisResult(
            analyzer_name=self.name,
            file_path=file or getattr(celldata, 'filePath', 'unknown'),
            success=True
        )
        if True:  # Uncomment to enable error handling
        #try:
            # Update parameters with any kwargs
            if kwargs:
                self.update_parameters(**kwargs)
            
            # Parse input to get consistent data format
            if celldata is not None:
                data = celldata
            else:
                data = parse_user_input(x, y, c, file)
            
            # Convert parameters to param_dict for compatibility
            param_dict = self._parameters_to_dict()
            
            # Determine sweeps to analyze
            if selected_sweeps is None:
                if hasattr(data, 'sweepList'):
                    selected_sweeps = data.sweepList
                else:
                    selected_sweeps = [0]
            
            # Run the analysis using the existing individual analysis method
            analysis_results = self.run_individual_analysis(
                data, selected_sweeps, param_dict, 
                popup=None, show_rejected=False
            )
            
           
            
            
        #except Exception as e:
        #    result.add_error(f"Analysis failed: {str(e)}")
            
        return analysis_results
    
    def run_batch_analysis(self, folder_path, param_dict=None, protocol_name=None):
        """
        Run analysis on a folder of files.
        
        Args:
            folder_path: Path to folder containing files
            param_dict: Analysis parameters
            protocol_name: Protocol filter
            
        Returns:
            tuple: (dataframes, summary_data) - format depends on analysis type
        """
        if isinstance(folder_path, str) or not isinstance(folder_path, list):
            filelist = glob.glob(folder_path + "/**/*.abf", recursive=True)
        elif isinstance(folder_path, list):
            #check if the folder_path are strings or cellData objects
            if isinstance(folder_path[0], str):
                filelist = folder_path
            elif isinstance(folder_path[0], cellData):
                filelist = [x.file for x in folder_path]
            elif isinstance(folder_path[0], pyabf.ABF):
                filelist = [x.name for x in folder_path]
            elif isinstance(folder_path[0], np.ndarray):
                filelist = folder_path
        else:
            logger.error('Folder_path must be a list of strings, a string, or a list of cellData objects')
            return None, None, None
        
        #run the feature extractor
        n_jobs = param_dict.get('n_jobs', 1) if param_dict else 1
        if n_jobs > 1: #if we are using multiprocessing
            pool = mp.Pool(processes=n_jobs)
            res = [pool.apply(self.analyze, args=(file, param_dict, protocol_name)) for file in filelist]
            pool.close()
           
            pool.join()
        #if we are not using multiprocessing
        else:
            for f in filelist:
                res = self.analyze(file=f, param_dict=copy.deepcopy(param_dict), protocol_name=protocol_name)

        #now we combine the results, each is a analysis result
        results = AnalysisResult.concatenate(res)
        return results

    def _parameters_to_dict(self) -> dict:
        """Convert AnalysisParameters to dict for backward compatibility"""
        param_dict = {}
        
        # Add common parameters
        param_dict['start'] = self.get_parameter('start_time', 0.0).value
        param_dict['end'] = self.get_parameter('end_time', 0.0).value
        param_dict['filter'] = 0  # Default filter
        
        # Add extra parameters
        for key, param in self._parameters.extra_params.items():
            if hasattr(param, 'value'):
                param_dict[key] = param.value
            else:
                param_dict[key] = param
                
        return param_dict
    
    def _populate_result(self, result: AnalysisResult, analysis_results: dict, data) -> None:
        """
        Populate the AnalysisResult with data from run_individual_analysis.
        Override in subclasses for analysis-specific formatting.
        """
        # Add basic metadata
        result.metadata['sweep_count'] = getattr(data, 'sweepCount', 1)
        result.metadata['protocol'] = getattr(data, 'protocol', 'unknown')
        
        # This is a basic implementation - subclasses should override
        #just shove it through sweepwise
        if isinstance(analysis_results, dict):
            result.detailed_data = self._convert_data_to_summary(analysis_results.get('spike_df', {}))
            result.summary_data = analysis_results.get('spike_summary', pd.DataFrame())
            result.sweep_data = analysis_results.get('running_bin', pd.DataFrame())
        else:
            # If analysis_results is not a dict, assume it's a DataFrame
            result.detailed_data = analysis_results
            result.summary_data = pd.DataFrame()

        
    
    def _convert_data_to_summary(self, df_dict=None) -> pd.DataFrame:
        """
        Convert spike dataframe dictionary to summary format.
        Override in subclasses for specific formatting needs.
        """
        if df_dict is None or not isinstance(df_dict, dict) or not isinstance(df_dict, pd.DataFrame):
            logger.warning("No spike data provided or data is not a dictionary")
            return pd.DataFrame()

        # Simple implementation - just concatenate all sweeps
        all_spikes = []
        for sweep, df in df_dict.items():
            if not df.empty:
                df_copy = df.copy()
                df_copy['sweep'] = sweep
                all_spikes.append(df_copy)
                
        if all_spikes:
            return pd.concat(all_spikes, ignore_index=True)
        else:
            return pd.DataFrame()
        
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

