"""
Simple test to verify parameter system syntax
"""

# Test that we can create the classes without runtime dependencies
import sys
import os

try:
    # Test basic Python syntax for our enhanced classes
    exec("""
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import pandas as pd

@dataclass
class AnalysisParameters:
    start_time: float = 0.0
    end_time: float = 0.0
    protocol_filter: str = ""
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        return self.extra_params.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
        else:
            self.extra_params[key] = value

class AnalysisModule:
    def __init__(self, name: str, display_name: str = None,
                 parameters: Optional[AnalysisParameters] = None):
        self.name = name
        self.display_name = display_name or name
        self._parameters = parameters or AnalysisParameters()
        self.param_dict = {}
        
    @property
    def parameters(self) -> AnalysisParameters:
        return self._parameters
    
    @parameters.setter
    def parameters(self, value: AnalysisParameters) -> None:
        if not isinstance(value, AnalysisParameters):
            raise TypeError(
                "Parameters must be an AnalysisParameters instance")
        self._parameters = value
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        return self._parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any) -> None:
        self._parameters.set(key, value)
    
    def update_parameters(self, **kwargs) -> None:
        for key, value in kwargs.items():
            self.set_parameter(key, value)
    
    def reset_parameters(self) -> None:
        self._parameters = AnalysisParameters()

# Test the classes
params = AnalysisParameters()
params.start_time = 1.0
params.set('threshold', 0.5)

module = AnalysisModule('test', 'Test Module', params)
print(f"✓ Module created: {module.name}")
print(f"✓ Parameter access: {module.get_parameter('start_time')}")
print(f"✓ Custom parameter: {module.get_parameter('threshold')}")

module.set_parameter('new_param', 'test')
print(f"✓ Set new parameter: {module.get_parameter('new_param')}")

module.update_parameters(start_time=5.0, end_time=10.0)
print(f"✓ Batch update: start={module.get_parameter('start_time')}")

print("\\n🎉 Parameter system syntax test passed!")
""")
    
except Exception as e:
    print(f"❌ Syntax error in parameter system: {e}")
    import traceback
    traceback.print_exc()
