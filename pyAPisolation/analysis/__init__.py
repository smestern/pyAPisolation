# Analysis Framework for pyAPisolation
# 
# This package provides a modular analysis framework for electrophysiology data analysis.
# It includes:
# - Abstract base classes for analysis modules
# - Registry system for managing analysis types
# - Built-in analysis modules (spike, subthreshold)
# - Utilities for easy module registration

from .base import AnalysisModule
from .registry import AnalysisRegistry, analysis_registry
from .builtin_modules import SpikeAnalysisModule, SubthresholdAnalysisModule
from .utilities import (
    register_analysis_module,
    register_analysis_with_tab,
    list_available_analyses,
    get_analysis_module,
    analysis_module
)

__all__ = [
    # Core classes
    'AnalysisModule',
    'AnalysisRegistry',
    'analysis_registry',
    
    # Built-in modules
    'SpikeAnalysisModule', 
    'SubthresholdAnalysisModule',
    
    # Utility functions
    'register_analysis_module',
    'register_analysis_with_tab', 
    'list_available_analyses',
    'get_analysis_module',
    'analysis_module'
]
