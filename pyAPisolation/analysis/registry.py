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
    
    def get_analyzer(self, name: str) -> Optional[AnalysisModule]:
        """
        Get an analyzer module by name.
        
        This is a convenience method that is equivalent to get_module.
        """
        return self.get_module(name)

# Global registry instance
registry = AnalysisRegistry()
