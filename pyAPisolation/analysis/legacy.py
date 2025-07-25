"""
Legacy wrapper for backward compatibility

This module provides wrappers to maintain compatibility with existing
code while using the new modular analysis framework.
"""

from typing import Any, Dict, Optional, Tuple
import pandas as pd

from .registry import AnalysisRegistry
from .base import AnalysisParameters
from ..featureExtractor import (
    process_file as legacy_process_file,
    analyze_subthres as legacy_analyze_subthres,
    save_data_frames as legacy_save_data_frames,
    save_subthres_data as legacy_save_subthres_data
)


class LegacyAnalysisWrapper:
    """Wrapper to maintain compatibility with legacy analysis functions"""
    
    def __init__(self, registry: Optional[AnalysisRegistry] = None):
        self.registry = registry or AnalysisRegistry()
    
    def process_file(self, file_path: str, param_dict: Dict[str, Any],
                    protocol_name: str = "") -> Tuple[pd.DataFrame, 
                                                     pd.DataFrame, 
                                                     pd.DataFrame]:
        """
        Legacy process_file wrapper using new spike analyzer
        
        Args:
            file_path: Path to ABF file
            param_dict: Legacy parameter dictionary
            protocol_name: Protocol filter
            
        Returns:
            Tuple of (spike_count_df, full_df, running_bin_df)
        """
        # Convert legacy parameters
        parameters = self._convert_legacy_spike_params(param_dict)
        parameters.protocol_filter = protocol_name
        
        # Use new spike analyzer
        analyzer = self.registry.get_analyzer('spike')
        result = analyzer.analyze_file(file_path, parameters)
        
        if result.success:
            return (
                result.summary_data or pd.DataFrame(),
                result.detailed_data or pd.DataFrame(),
                result.sweep_data or pd.DataFrame()
            )
        else:
            # Return empty dataframes on failure
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    def analyze_subthres(self, abf, **kwargs) -> Tuple[pd.DataFrame,
                                                      pd.DataFrame]:
        """
        Legacy analyze_subthres wrapper using new subthreshold analyzer
        
        Args:
            abf: ABF object or file path
            **kwargs: Legacy keyword arguments
            
        Returns:
            Tuple of (sweep_df, avg_df)
        """
        # Convert legacy parameters
        parameters = self._convert_legacy_subthres_params(kwargs)
        
        # Handle ABF object vs file path
        if hasattr(abf, 'abfFilePath'):
            file_path = abf.abfFilePath
        else:
            file_path = str(abf)
        
        # Use new subthreshold analyzer
        analyzer = self.registry.get_analyzer('subthreshold')
        result = analyzer.analyze_file(file_path, parameters)
        
        if result.success:
            return (
                result.detailed_data or pd.DataFrame(),
                result.summary_data or pd.DataFrame()
            )
        else:
            return pd.DataFrame(), pd.DataFrame()
    
    def save_data_frames(self, *args, **kwargs):
        """Legacy save_data_frames wrapper"""
        return legacy_save_data_frames(*args, **kwargs)
    
    def save_subthres_data(self, *args, **kwargs):
        """Legacy save_subthres_data wrapper"""
        return legacy_save_subthres_data(*args, **kwargs)
    
    def _convert_legacy_spike_params(self, param_dict: Dict[str, Any]
                                   ) -> AnalysisParameters:
        """Convert legacy spike parameter dictionary to AnalysisParameters"""
        parameters = AnalysisParameters()
        
        # Map legacy parameter names
        param_mapping = {
            'start': 'start_time',
            'end': 'end_time',
            'dv_cutoff': 'dv_cutoff',
            'max_interval': 'max_interval',
            'min_height': 'min_height',
            'min_peak': 'min_peak',
            'thresh_frac': 'thresh_frac',
            'filter': 'filter',
            'bessel_filter': 'bessel_filter',
            'stim_find': 'stim_find'
        }
        
        for legacy_key, new_key in param_mapping.items():
            if legacy_key in param_dict:
                if new_key in ['start_time', 'end_time']:
                    setattr(parameters, new_key, param_dict[legacy_key])
                else:
                    parameters.extra_params[new_key] = param_dict[legacy_key]
        
        return parameters
    
    def _convert_legacy_subthres_params(self, kwargs: Dict[str, Any]
                                      ) -> AnalysisParameters:
        """Convert legacy subthreshold kwargs to AnalysisParameters"""
        parameters = AnalysisParameters()
        
        # Map legacy parameter names
        param_mapping = {
            'protocol_name': 'protocol_filter',
            'time_after': 'time_after',
            'subt_sweeps': 'subt_sweeps',
            'start_sear': 'start_sear',
            'end_sear': 'end_sear',
            'savfilter': 'savfilter',
            'bplot': 'bplot'
        }
        
        for legacy_key, new_key in param_mapping.items():
            if legacy_key in kwargs:
                if new_key == 'protocol_filter':
                    parameters.protocol_filter = kwargs[legacy_key]
                else:
                    parameters.extra_params[new_key] = kwargs[legacy_key]
        
        return parameters


# Create global instance for backward compatibility
_legacy_wrapper = LegacyAnalysisWrapper()

# Export legacy functions
process_file = _legacy_wrapper.process_file
analyze_subthres = _legacy_wrapper.analyze_subthres
save_data_frames = _legacy_wrapper.save_data_frames
save_subthres_data = _legacy_wrapper.save_subthres_data
