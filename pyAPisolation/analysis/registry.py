"""
Registry for managing available analyzers

This module provides a registry system for discovering and managing
different types of analyzers.
"""

from typing import Dict, Type, List, Optional
from .base import BaseAnalyzer


class AnalysisRegistry:
    """Registry for managing available analyzers"""
    
    def __init__(self):
        self._analyzers: Dict[str, Type[BaseAnalyzer]] = {}
        self._instances: Dict[str, BaseAnalyzer] = {}
    
    def register(self, name: str, analyzer_class: Type[BaseAnalyzer]) -> None:
        """
        Register an analyzer class
        
        Args:
            name: Name to register the analyzer under
            analyzer_class: Analyzer class to register
        """
        if not issubclass(analyzer_class, BaseAnalyzer):
            raise ValueError(f"{analyzer_class} must inherit from BaseAnalyzer")
        
        self._analyzers[name] = analyzer_class
        # Clear cached instance if it exists
        if name in self._instances:
            del self._instances[name]
    
    def get_analyzer(self, name: str) -> BaseAnalyzer:
        """
        Get an analyzer instance by name
        
        Args:
            name: Name of the analyzer
            
        Returns:
            Analyzer instance
            
        Raises:
            KeyError: If analyzer is not registered
        """
        if name not in self._analyzers:
            raise KeyError(f"Analyzer '{name}' not registered. "
                          f"Available: {list(self._analyzers.keys())}")
        
        # Use cached instance if available
        if name not in self._instances:
            self._instances[name] = self._analyzers[name](name)
        
        return self._instances[name]
    
    def list_analyzers(self) -> List[str]:
        """Return list of registered analyzer names"""
        return list(self._analyzers.keys())
    
    def unregister(self, name: str) -> None:
        """
        Unregister an analyzer
        
        Args:
            name: Name of analyzer to unregister
        """
        if name in self._analyzers:
            del self._analyzers[name]
        if name in self._instances:
            del self._instances[name]
    
    def get_analyzer_info(self, name: str) -> Dict[str, str]:
        """
        Get information about a registered analyzer
        
        Args:
            name: Name of the analyzer
            
        Returns:
            Dictionary with analyzer information
        """
        if name not in self._analyzers:
            raise KeyError(f"Analyzer '{name}' not registered")
        
        analyzer = self.get_analyzer(name)
        return {
            'name': name,
            'type': analyzer.analysis_type,
            'class': self._analyzers[name].__name__,
            'module': self._analyzers[name].__module__
        }
