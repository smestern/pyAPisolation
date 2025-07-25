"""
Spike analysis implementation

This module provides spike detection and feature extraction using the
existing ipfx-based infrastructure while conforming to the new modular
architecture.
"""

import os
import copy
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import pyabf
from ipfx.feature_extractor import SpikeFeatureExtractor

from .base import BaseAnalyzer, AnalysisResult, AnalysisParameters
from ..featureExtractor import process_file as legacy_process_file


class SpikeAnalyzer(BaseAnalyzer):
    """Analyzer for spike detection and feature extraction"""
    
    def __init__(self, name: str = "spike"):
        super().__init__(name)
        self._setup_default_parameters()
    
    @property
    def analysis_type(self) -> str:
        return "spike"
    
    def _setup_default_parameters(self) -> None:
        """Setup default parameters for spike analysis"""
        self._default_params.extra_params.update({
            'dv_cutoff': 7.0,           # mV/s threshold for spike detection
            'max_interval': 0.010,      # max time from threshold to peak (s)
            'min_height': 2.0,          # min threshold-to-peak height (mV)
            'min_peak': -10.0,          # min peak voltage (mV)
            'thresh_frac': 0.05,        # fraction of max dV/dt for threshold
            'filter': 0,                # ipfx filter setting
            'bessel_filter': 0.0,       # bessel filter frequency (0 = no filter)
            'stim_find': False,         # search based on stimulus
        })
    
    def validate_parameters(self, parameters: AnalysisParameters) -> List[str]:
        """Validate spike analysis parameters"""
        errors = []
        
        # Check required numeric parameters
        numeric_params = [
            'dv_cutoff', 'max_interval', 'min_height', 
            'min_peak', 'thresh_frac'
        ]
        
        for param in numeric_params:
            value = parameters.get(param)
            if value is None:
                errors.append(f"Missing required parameter: {param}")
            elif not isinstance(value, (int, float)):
                errors.append(f"Parameter {param} must be numeric")
        
        # Validate ranges
        if parameters.get('thresh_frac', 0) <= 0 or parameters.get('thresh_frac', 0) > 1:
            errors.append("thresh_frac must be between 0 and 1")
        
        if parameters.get('max_interval', 0) <= 0:
            errors.append("max_interval must be positive")
        
        return errors
    
    def analyze_file(self, file_path: str, 
                    parameters: AnalysisParameters) -> AnalysisResult:
        """
        Analyze spikes in a single ABF file
        
        Args:
            file_path: Path to ABF file
            parameters: Analysis parameters
            
        Returns:
            AnalysisResult with spike analysis data
        """
        result = AnalysisResult(
            analyzer_name=self.name,
            file_path=file_path,
            success=False
        )
        
        try:
            # Validate parameters
            validation_errors = self.validate_parameters(parameters)
            if validation_errors:
                for error in validation_errors:
                    result.add_error(error)
                return result
            
            # Convert parameters to legacy format
            param_dict = self._convert_parameters(parameters)
            
            # Use legacy processing function
            spike_count_df, full_df, running_bin_df = legacy_process_file(
                file_path, param_dict, parameters.protocol_filter
            )
            
            # Store results
            result.summary_data = spike_count_df
            result.detailed_data = full_df
            result.sweep_data = running_bin_df
            
            # Add metadata
            result.metadata.update({
                'file_name': os.path.basename(file_path),
                'parameters_used': param_dict,
                'total_sweeps': len(full_df) if full_df is not None else 0
            })
            
            result.success = True
            
        except Exception as e:
            result.add_error(f"Analysis failed: {str(e)}")
        
        return result
    
    def analyze_sweep(self, sweep_data: Dict[str, np.ndarray], 
                     parameters: AnalysisParameters) -> Dict[str, Any]:
        """
        Analyze spikes in a single sweep
        
        Args:
            sweep_data: Dictionary with 'time', 'voltage', 'current' arrays
            parameters: Analysis parameters
            
        Returns:
            Dictionary of spike features for this sweep
        """
        # Convert parameters
        param_dict = self._convert_parameters(parameters)
        
        # Create spike extractor
        extractor = SpikeFeatureExtractor(
            filter=param_dict['filter'],
            dv_cutoff=param_dict['dv_cutoff'],
            max_interval=param_dict['max_interval'],
            min_height=param_dict['min_height'],
            min_peak=param_dict['min_peak'],
            start=param_dict['start'],
            end=param_dict['end'],
            thresh_frac=param_dict['thresh_frac']
        )
        
        # Extract features
        spike_df = extractor.process(
            sweep_data['time'],
            sweep_data['voltage'], 
            sweep_data['current']
        )
        
        # Convert to dictionary format
        if spike_df.empty:
            return {'spike_count': 0, 'features': {}}
        
        return {
            'spike_count': len(spike_df),
            'features': spike_df.to_dict('records')
        }
    
    def _convert_parameters(self, parameters: AnalysisParameters) -> Dict[str, Any]:
        """Convert AnalysisParameters to legacy parameter dictionary"""
        param_dict = {
            'filter': parameters.get('filter', 0),
            'dv_cutoff': parameters.get('dv_cutoff', 7.0),
            'start': parameters.start_time,
            'end': parameters.end_time,
            'max_interval': parameters.get('max_interval', 0.010),
            'min_height': parameters.get('min_height', 2.0),
            'min_peak': parameters.get('min_peak', -10.0),
            'thresh_frac': parameters.get('thresh_frac', 0.05),
            'stim_find': parameters.get('stim_find', False),
            'bessel_filter': parameters.get('bessel_filter', 0.0)
        }
        return param_dict
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Return parameter schema for spike analysis"""
        base_schema = super().get_parameter_schema()
        
        spike_schema = {
            'dv_cutoff': {
                'type': 'float',
                'default': 7.0,
                'description': 'dV/dt threshold for spike detection (mV/s)',
                'min': 0.1,
                'max': 100.0
            },
            'max_interval': {
                'type': 'float',
                'default': 0.010,
                'description': 'Max time from threshold to peak (seconds)',
                'min': 0.001,
                'max': 0.1
            },
            'min_height': {
                'type': 'float',
                'default': 2.0,
                'description': 'Min threshold-to-peak height (mV)',
                'min': 0.1,
                'max': 50.0
            },
            'min_peak': {
                'type': 'float',
                'default': -10.0,
                'description': 'Min peak voltage (mV)',
                'min': -100.0,
                'max': 50.0
            },
            'thresh_frac': {
                'type': 'float',
                'default': 0.05,
                'description': 'Fraction of max dV/dt for threshold refinement',
                'min': 0.01,
                'max': 1.0
            },
            'bessel_filter': {
                'type': 'float',
                'default': 0.0,
                'description': 'Bessel filter frequency (Hz, 0=no filter)',
                'min': 0.0,
                'max': 10000.0
            },
            'stim_find': {
                'type': 'bool',
                'default': False,
                'description': 'Search for spikes based on stimulus timing'
            }
        }
        
        base_schema.update(spike_schema)
        return base_schema
