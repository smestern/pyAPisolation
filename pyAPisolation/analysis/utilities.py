"""
Utility functions for easy analysis module registration and management.

This module provides convenient functions and decorators for working with
the analysis module framework.
"""

from typing import Type, Any, Dict, Optional
from .base import AnalysisModule
from .registry import analysis_registry


def register_analysis_module(module_class: Type[AnalysisModule], 
                           *args, **kwargs) -> AnalysisModule:
    """
    Utility function to easily register a new analysis module.
    
    Args:
        module_class: The analysis module class (must inherit from AnalysisModule)
        *args, **kwargs: Arguments to pass to the module constructor
    
    Returns:
        The registered module instance
    
    Example:
        register_analysis_module(MyCustomAnalysisModule, param1="value")
    """
    try:
        # Create instance of the module
        module_instance = module_class(*args, **kwargs)
        
        # Register it
        analysis_registry.register_module(module_instance)
        
        return module_instance
        
    except Exception as e:
        print(f"Error registering module {module_class.__name__}: {e}")
        raise


def register_analysis_with_tab(module_class: Type[AnalysisModule], 
                              tab_index: int, *args, **kwargs) -> AnalysisModule:
    """
    Utility function to register a new analysis module and assign it to a tab.
    
    Args:
        module_class: The analysis module class
        tab_index: Tab index to assign the module to
        *args, **kwargs: Arguments to pass to the module constructor
    
    Returns:
        The registered module instance
    
    Example:
        register_analysis_with_tab(MyCustomAnalysisModule, 2, param1="value")
    """
    module_instance = register_analysis_module(module_class, *args, **kwargs)
    analysis_registry.add_tab_mapping(tab_index, module_instance.name)
    return module_instance


def list_available_analyses() -> Dict[str, str]:
    """
    Utility function to list all available analysis modules.
    
    Returns:
        dict: Dictionary of {module_name: display_name}
    """
    return analysis_registry.list_modules_detailed()


def get_analysis_module(name: str) -> Optional[AnalysisModule]:
    """
    Utility function to get an analysis module by name.
    
    Args:
        name: Name of the analysis module
    
    Returns:
        AnalysisModule or None if not found
    """
    return analysis_registry.get_module(name)


def analysis_module(name: str = None, display_name: str = None, 
                   tab_index: int = None):
    """
    Decorator to automatically register an analysis module.
    
    Args:
        name: Module name (defaults to class name lowercased)
        display_name: Display name (defaults to name)
        tab_index: Optional tab index to assign
    
    Example:
        @analysis_module(name="my_analysis", display_name="My Analysis", tab_index=2)
        class MyAnalysisModule(AnalysisModule):
            # ... implementation ...
    """
    def decorator(cls):
        # Determine module name
        module_name = name or cls.__name__.lower().replace('module', '').replace('analysis', '')
        
        # Create instance
        instance = cls()
        
        # Override name and display_name if provided
        if name:
            instance.name = name
        if display_name:
            instance.display_name = display_name
        
        # Register the module
        analysis_registry.register_module(instance)
        
        # Add tab mapping if requested
        if tab_index is not None:
            analysis_registry.add_tab_mapping(tab_index, instance.name)
        
        return cls
    
    return decorator


def unregister_analysis_module(name: str):
    """
    Utility function to unregister an analysis module.
    
    Args:
        name: Name of the module to unregister
    """
    analysis_registry.unregister_module(name)


def get_registry_info() -> Dict[str, Any]:
    """
    Get detailed information about the current registry state.
    
    Returns:
        dict: Registry information including modules and tab mappings
    """
    return analysis_registry.get_registry_info()


def clear_all_modules():
    """
    Clear all registered modules (useful for testing).
    Note: This will also clear built-in modules.
    """
    analysis_registry.clear_registry()


# Initialize built-in modules when this module is imported
def _initialize_builtin_modules():
    """Initialize the built-in analysis modules"""
    try:
        from .builtin_modules import SpikeAnalysisModule, SubthresholdAnalysisModule
        
        # Register built-in modules
        spike_module = SpikeAnalysisModule()
        subthres_module = SubthresholdAnalysisModule()
        
        analysis_registry.register_module(spike_module)
        analysis_registry.register_module(subthres_module)
        
        # Set up legacy tab mapping
        analysis_registry.add_tab_mapping(0, 'spike')
        analysis_registry.add_tab_mapping(1, 'subthres')
        
    except ImportError as e:
        print(f"Warning: Could not initialize built-in modules: {e}")


# Auto-initialize when module is imported
_initialize_builtin_modules()
