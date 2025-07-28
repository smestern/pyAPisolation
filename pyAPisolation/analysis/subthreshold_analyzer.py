"""
Subthreshold analysis implementation

This module provides subthreshold feature analysis using the existing
patch_subthres infrastructure while conforming to the new modular
architecture.
"""

import os
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from .base import BaseAnalyzer, AnalysisResult, AnalysisParameters
from ..featureExtractor import analyze_subthres, preprocess_abf_subthreshold


class SubthresholdAnalyzer(BaseAnalyzer):
    """Analyzer for subthreshold membrane properties"""
    
    def __init__(self, name: str = "subthreshold"):
        super().__init__(name)
        self._setup_default_parameters()
    
    @property
    def analysis_type(self) -> str:
        return "subthreshold"
    
    def _setup_default_parameters(self) -> None:
        """Setup default parameters for subthreshold analysis"""
        self._default_params.extra_params.update({
            'time_after': 50.0,         # time after current step (%)
            'subt_sweeps': None,        # specific sweeps to analyze
            'start_sear': None,         # start time for analysis
            'end_sear': None,           # end time for analysis
            'savfilter': 0,             # savitzky-golay filter window
            'bplot': False,             # generate plots
        })
    
    def validate_parameters(self, parameters: AnalysisParameters) -> List[str]:
        """Validate subthreshold analysis parameters"""
        errors = []
        
        # Check time_after range
        time_after = parameters.get('time_after', 50.0)
        if not isinstance(time_after, (int, float)):
            errors.append("time_after must be numeric")
        elif time_after <= 0 or time_after > 100:
            errors.append("time_after must be between 0 and 100 percent")
        
        # Check sweep specification if provided
        subt_sweeps = parameters.get('subt_sweeps')
        if subt_sweeps is not None:
            if not isinstance(subt_sweeps, (list, np.ndarray)):
                errors.append("subt_sweeps must be a list or array of integers")
        
        # Check time bounds
        start_sear = parameters.get('start_sear')
        end_sear = parameters.get('end_sear')
        
        if start_sear is not None and not isinstance(start_sear, (int, float)):
            errors.append("start_sear must be numeric")
        
        if end_sear is not None and not isinstance(end_sear, (int, float)):
            errors.append("end_sear must be numeric")
        
        if (start_sear is not None and end_sear is not None and 
            start_sear >= end_sear):
            errors.append("start_sear must be less than end_sear")
        
        return errors
    
    def analyze_file(self, file_path: str,
                    parameters: AnalysisParameters) -> AnalysisResult:
        """
        Analyze subthreshold properties in a single ABF file
        
        Args:
            file_path: Path to ABF file
            parameters: Analysis parameters
            
        Returns:
            AnalysisResult with subthreshold analysis data
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
            
            # Use legacy analysis function directly for full ABF files
            if parameters.protocol_filter:
                # Use preprocessing function if protocol filter is specified
                sweep_df, avg_df = preprocess_abf_subthreshold(
                    file_path, 
                    parameters.protocol_filter,
                    self._convert_parameters(parameters)
                )
            else:
                # Load ABF and analyze directly
                import pyabf
                abf = pyabf.ABF(file_path)
                sweep_df, avg_df = analyze_subthres(
                    abf, **self._convert_parameters(parameters)
                )
            
            # Store results
            result.summary_data = avg_df
            result.detailed_data = sweep_df
            
            # Add metadata
            result.metadata.update({
                'file_name': os.path.basename(file_path),
                'parameters_used': self._convert_parameters(parameters),
                'total_sweeps': len(sweep_df) if sweep_df is not None else 0
            })
            
            result.success = True
            
        except Exception as e:
            result.add_error(f"Subthreshold analysis failed: {str(e)}")
        
        return result
    
    def analyze_sweep(self, sweep_data: Dict[str, np.ndarray],
                     parameters: AnalysisParameters) -> Dict[str, Any]:
        """
        Analyze subthreshold properties in a single sweep
        
        Note: This is a simplified implementation. Full subthreshold analysis
        typically requires multiple sweeps with different current injections.
        """
        # Import required functions
        from ..patch_subthres import (
            membrane_resistance, exp_decay_factor, compute_sag, rmp_mode
        )
        
        time_data = sweep_data['time']
        voltage_data = sweep_data['voltage']
        current_data = sweep_data['current']
        
        time_after = parameters.get('time_after', 50.0)
        
        try:
            # Basic membrane properties
            resistance = membrane_resistance(time_data, voltage_data, current_data)
            
            # Exponential decay analysis
            tau1, tau2, curve, r2_2p, r2_1p, tau_1p = exp_decay_factor(
                time_data, voltage_data, current_data, time_after
            )
            
            # Voltage sag
            sag_diff, min_voltage = compute_sag(
                time_data, voltage_data, current_data, time_after
            )
            
            # Resting membrane potential
            rmp = rmp_mode(voltage_data, current_data)
            
            return {
                'membrane_resistance': resistance / 1e9 if resistance else np.nan,  # GOhm
                'tau_fast': tau1 if tau1 else np.nan,
                'tau_slow': tau2 if tau2 else np.nan,
                'tau_single': tau_1p if tau_1p else np.nan,
                'r_squared_2p': r2_2p if r2_2p else np.nan,
                'r_squared_1p': r2_1p if r2_1p else np.nan,
                'voltage_sag': sag_diff if sag_diff else np.nan,
                'min_voltage': min_voltage if min_voltage else np.nan,
                'resting_potential': rmp if rmp else np.nan
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'membrane_resistance': np.nan,
                'tau_fast': np.nan,
                'tau_slow': np.nan
            }
    
    def _convert_parameters(self, parameters: AnalysisParameters) -> Dict[str, Any]:
        """Convert AnalysisParameters to legacy parameter dictionary"""
        param_dict = {
            'protocol_name': parameters.protocol_filter,
            'time_after': parameters.get('time_after', 50.0),
            'subt_sweeps': parameters.get('subt_sweeps'),
            'start_sear': parameters.get('start_sear'),
            'end_sear': parameters.get('end_sear'),
            'savfilter': parameters.get('savfilter', 0),
            'bplot': parameters.get('bplot', False)
        }
        return param_dict
    
    def get_parameter_schema(self) -> Dict[str, Any]:
        """Return parameter schema for subthreshold analysis"""
        base_schema = super().get_parameter_schema()
        
        subthres_schema = {
            'time_after': {
                'type': 'float',
                'default': 50.0,
                'description': 'Percentage of decay to analyze after step',
                'min': 1.0,
                'max': 100.0
            },
            'subt_sweeps': {
                'type': 'list',
                'default': None,
                'description': 'Specific sweep numbers to analyze (None=auto)',
                'optional': True
            },
            'start_sear': {
                'type': 'float',
                'default': None,
                'description': 'Start time for analysis window (s)',
                'optional': True
            },
            'end_sear': {
                'type': 'float',
                'default': None,
                'description': 'End time for analysis window (s)',
                'optional': True
            },
            'savfilter': {
                'type': 'int',
                'default': 0,
                'description': 'Savitzky-Golay filter window (0=no filter)',
                'min': 0,
                'max': 51
            },
            'bplot': {
                'type': 'bool',
                'default': False,
                'description': 'Generate diagnostic plots'
            }
        }
        
        base_schema.update(subthres_schema)
        return base_schema

class ResistanceLadder(BaseAnalyzer):
    """
    Compute the I-V curve resistance ladder for subthreshold sweeps.

    This is a specialized analyzer that computes the resistance ladder
    """
    def __init__(self, name: str = "resistance_ladder"):
        super().__init__(name)
        self._setup_default_parameters()

    def _setup_default_parameters(self) -> None:
        """Setup default parameters for resistance ladder analysis"""
        self._default_params.extra_params.update({
            'sweep_range': None,  # Range of sweeps to analyze
            'voltage_range': None,  # Voltage range for I-V curve
            'current_range': None,  # Current range for I-V curve
            'start_time': 0.0,  # Start time for analysis
            'end_time': None,  # End time for analysis
            'i_channel': 0,  # Channel to analyze
            'v_channel': 1,  # Channel to analyze
            'plot': True  # Generate plot of resistance ladder
        })
    def analyze_file(self, file_path: str,
                     parameters: AnalysisParameters) -> AnalysisResult:
        """ Analyze resistance ladder in a single ABF file
        
        Args:
            file_path: Path to ABF file
            parameters: Analysis parameters
            
        Returns:
            AnalysisResult with resistance ladder data
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
            
            # Load ABF and compute resistance ladder
            import pyabf
            abf = pyabf.ABF(file_path)
            
            # Call the resistance ladder analysis function
            ladder_df = patch_utils.compute_resistance_ladder(
                abf, **self._convert_parameters(parameters)
            )
            
            # Store results
            result.summary_data = ladder_df
            
            # Add metadata
            result.metadata.update({
                'file_name': os.path.basename(file_path),
                'parameters_used': self._convert_parameters(parameters),
                'total_sweeps': len(ladder_df) if ladder_df is not None else 0
            })
            
            result.success = True
            
        except Exception as e:
            result.add_error(f"Resistance ladder analysis failed: {str(e)}")
        
        return result
    