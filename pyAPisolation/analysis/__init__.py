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

from .base import BaseAnalyzer, AnalysisResult, AnalysisParameters
from .spike_analyzer import SpikeAnalyzer
from .subthreshold_analyzer import SubthresholdAnalyzer, ResistanceLadder
from .registry import AnalysisRegistry
from .runner import AnalysisRunner
from .legacy import LegacyAnalysisWrapper

# Initialize the registry with built-in analyzers
registry = AnalysisRegistry()
registry.register('spike', SpikeAnalyzer)
registry.register('subthreshold', SubthresholdAnalyzer)
registry.register('resistance_ladder', ResistanceLadder)

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
