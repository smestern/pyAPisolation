"""
Base classes for the analysis framework

This module defines the core interfaces and data structures that all
analyzers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np


@dataclass
class AnalysisParameters:
    """Parameters for analysis configuration"""
    
    # Common parameters
    start_time: float = 0.0
    end_time: float = 0.0
    protocol_filter: str = ""
    
    # Additional parameters stored as dict for flexibility
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
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
            self.extra_params[key] = value


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


class BaseAnalyzer(ABC):
    """Abstract base class for all analyzers"""
    
    def __init__(self, name: str):
        self.name = name
        self._default_params = AnalysisParameters()
    
    @property
    @abstractmethod
    def analysis_type(self) -> str:
        """Return the type of analysis this analyzer performs"""
        pass
    
    @property
    def default_parameters(self) -> AnalysisParameters:
        """Return default parameters for this analyzer"""
        return self._default_params
    
    @abstractmethod
    def analyze_file(self, file_path: str, 
                    parameters: AnalysisParameters) -> AnalysisResult:
        """
        Analyze a single file
        
        Args:
            file_path: Path to the file to analyze
            parameters: Analysis parameters
            
        Returns:
            AnalysisResult containing the results
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: AnalysisParameters) -> List[str]:
        """
        Validate analysis parameters
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    def analyze_sweep(self, sweep_data: Dict[str, np.ndarray], 
                     parameters: AnalysisParameters) -> Dict[str, Any]:
        """
        Analyze a single sweep (optional implementation)
        
        Args:
            sweep_data: Dictionary with 'time', 'voltage', 'current' arrays
            parameters: Analysis parameters
            
        Returns:
            Dictionary of analysis results for this sweep
        """
        raise NotImplementedError(
            f"{self.name} does not support single-sweep analysis"
        )
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """
        Return a schema describing the parameters this analyzer accepts
        
        Returns:
            Dictionary describing parameter names, types, defaults, etc.
        """
        return {
            'start_time': {
                'type': 'float',
                'default': 0.0,
                'description': 'Start time for analysis (seconds)'
            },
            'end_time': {
                'type': 'float', 
                'default': 0.0,
                'description': 'End time for analysis (seconds, 0 for end)'
            },
            'protocol_filter': {
                'type': 'str',
                'default': '',
                'description': 'Protocol name filter'
            }
        }
