"""
Registry system for managing analysis modules.

This module provides the AnalysisRegistry class and global registry instance
for managing and accessing analysis modules.
"""

from typing import Dict, Optional
from .base import AnalysisModule


class AnalysisRegistry:
    """
    Registry for managing analysis modules
    """
    
    def __init__(self):
        self.modules: Dict[str, AnalysisModule] = {}
        self.tab_mapping: Dict[int, str] = {}
    
    def register_module(self, module: AnalysisModule):
        """Register a new analysis module"""
        if not isinstance(module, AnalysisModule):
            raise TypeError(
                f"Module must be an instance of AnalysisModule, "
                f"got {type(module)}"
            )
        
        if module.name in self.modules:
            print(f"Warning: Overwriting existing module '{module.name}'")
        
        self.modules[module.name] = module
        print(f"Registered analysis module: {module.display_name} "
              f"('{module.name}')")
    
    def get_module(self, name: str) -> Optional[AnalysisModule]:
        """Get an analysis module by name"""
        return self.modules.get(name)
    
    def get_module_by_tab(self, tab_index: int) -> Optional[AnalysisModule]:
        """Get analysis module by tab index (for legacy compatibility)"""
        module_name = self.tab_mapping.get(tab_index)
        return self.get_module(module_name) if module_name else None
    
    def list_modules(self):
        """List all registered modules"""
        return list(self.modules.keys())
    
    def list_modules_detailed(self):
        """List all registered modules with detailed information"""
        return {name: module.display_name 
                for name, module in self.modules.items()}
    
    def add_tab_mapping(self, tab_index: int, module_name: str):
        """Add a new tab mapping"""
        if module_name not in self.modules:
            raise ValueError(f"Module '{module_name}' is not registered")
        
        if tab_index in self.tab_mapping:
            old_module = self.tab_mapping[tab_index]
            print(f"Warning: Tab {tab_index} already mapped to "
                  f"'{old_module}', overwriting with '{module_name}'")
        
        self.tab_mapping[tab_index] = module_name
        print(f"Mapped tab {tab_index} to module '{module_name}'")
    
    def unregister_module(self, name: str):
        """Unregister an analysis module"""
        if name in self.modules:
            del self.modules[name]
            # Remove from tab mappings
            tabs_to_remove = [tab for tab, module_name in self.tab_mapping.items()
                             if module_name == name]
            for tab in tabs_to_remove:
                del self.tab_mapping[tab]
            print(f"Unregistered module '{name}'")
        else:
            print(f"Module '{name}' was not registered")
    
    def clear_registry(self):
        """Clear all registered modules and tab mappings"""
        self.modules.clear()
        self.tab_mapping.clear()
        print("Cleared all registered modules")
    
    def get_registry_info(self):
        """Get detailed information about the registry state"""
        return {
            'modules': self.list_modules_detailed(),
            'tab_mappings': dict(self.tab_mapping),
            'module_count': len(self.modules)
        }


# Global registry instance - will be populated with built-in modules
analysis_registry = AnalysisRegistry()

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
