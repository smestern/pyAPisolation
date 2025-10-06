"""
Core analysis module for pyAPisolation

This module provides a modular, extensible framework for electrophysiology analysis.
It separates analysis logic from GUI code and enables easy CLI access.

Key components:
- BaseAnalyzer: Abstract base class for all analyzers
- SpikeAnalyzer: Spike detection and feature extraction
- SubthresholdAnalyzer: Subthreshold feature analysis
- AnalysisRegistry: Registry for available analyzers
- AnalysisRunner: Main runner for batch processing
"""
print(" ==== THIS MODULE IS UNDER ACTIVE DEVELOPMENT, API MAY CHANGE. PLEASE USE WITH CAUTION ==== ")
from .base import AnalysisResult, AnalysisParameters

from .registry import AnalysisRegistry, registry
from .runner import AnalysisRunner
#from .legacy import LegacyAnalysisWrapper
from .builtin_modules import SpikeAnalysisModule, SubthresholdAnalysisModule, ResistanceLadder

# Initialize the registry with built-in analyzers

registry.register_module(SpikeAnalysisModule())
registry.register_module(SubthresholdAnalysisModule())
registry.register_module(ResistanceLadder())

__all__ = [
    'BaseAnalyzer',
    'AnalysisResult', 
    'AnalysisParameters',
    'SpikeAnalyzer',
    'SubthresholdAnalyzer',
    'AnalysisRegistry',
    'AnalysisRunner',
    'LegacyAnalysisWrapper',
    'registry'
]
