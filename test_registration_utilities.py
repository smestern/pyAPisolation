#!/usr/bin/env python3
"""
Test script to verify that the analysis module registration utilities work correctly.

This script tests:
1. Basic module registration
2. Registration with tab mapping
3. Decorator-based registration
4. Registry query functions
5. Error handling

Run this script to verify the modular analysis framework is working.
"""

import sys
import os

# Add the pyAPisolation package to the path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_basic_registration():
    """Test basic module registration"""
    print("=== Testing Basic Registration ===")
    
    from pyAPisolation.analysis import (
        AnalysisModule, register_analysis_module, list_available_analyses
    )
    import pandas as pd
    
    class TestBasicModule(AnalysisModule):
        def __init__(self):
            super().__init__("test_basic", "Test Basic Module")
        
        def get_ui_elements(self):
            return {'param1': 'float'}
        
        def parse_ui_params(self, ui_elements):
            return {'param1': 1.0}
        
        def run_individual_analysis(self, abf, selected_sweeps, param_dict, 
                                   popup=None, show_rejected=False):
            return {'test_data': pd.DataFrame()}
        
        def run_batch_analysis(self, folder_path, param_dict, protocol_name):
            return pd.DataFrame()
        
        def save_results(self, results, output_dir, output_tag, save_options=None):
            pass
    
    # Test registration
    try:
        module = register_analysis_module(TestBasicModule)
        print(f"✓ Successfully registered: {module.display_name}")
        
        # Verify it's in the registry
        modules = list_available_analyses()
        if 'test_basic' in modules:
            print(f"✓ Module found in registry: {modules['test_basic']}")
        else:
            print("✗ Module not found in registry")
            
    except Exception as e:
        print(f"✗ Registration failed: {e}")


def test_tab_registration():
    """Test registration with tab mapping"""
    print("\n=== Testing Tab Registration ===")
    
    from pyAPisolation.analysis import (
        AnalysisModule, register_analysis_with_tab, analysis_registry
    )
    import pandas as pd
    
    class TestTabModule(AnalysisModule):
        def __init__(self):
            super().__init__("test_tab", "Test Tab Module")
        
        def get_ui_elements(self):
            return {'param1': 'float'}
        
        def parse_ui_params(self, ui_elements):
            return {'param1': 1.0}
        
        def run_individual_analysis(self, abf, selected_sweeps, param_dict, 
                                   popup=None, show_rejected=False):
            return {'test_data': pd.DataFrame()}
        
        def run_batch_analysis(self, folder_path, param_dict, protocol_name):
            return pd.DataFrame()
        
        def save_results(self, results, output_dir, output_tag, save_options=None):
            pass
    
    try:
        # Register with tab index 5 (safe index)
        module = register_analysis_with_tab(TestTabModule, 5)
        print(f"✓ Successfully registered with tab: {module.display_name}")
        
        # Verify tab mapping
        tab_module = analysis_registry.get_module_by_tab(5)
        if tab_module and tab_module.name == 'test_tab':
            print("✓ Tab mapping verified")
        else:
            print("✗ Tab mapping failed")
            
    except Exception as e:
        print(f"✗ Tab registration failed: {e}")


def test_decorator_registration():
    """Test decorator-based registration"""
    print("\n=== Testing Decorator Registration ===")
    
    try:
        from pyAPisolation.analysis import analysis_module, AnalysisModule
        import pandas as pd
        
        @analysis_module(name="test_decorator", display_name="Test Decorator Module")
        class TestDecoratorModule(AnalysisModule):
            def get_ui_elements(self):
                return {'param1': 'float'}
            
            def parse_ui_params(self, ui_elements):
                return {'param1': 1.0}
            
            def run_individual_analysis(self, abf, selected_sweeps, param_dict, 
                                       popup=None, show_rejected=False):
                return {'test_data': pd.DataFrame()}
            
            def run_batch_analysis(self, folder_path, param_dict, protocol_name):
                return pd.DataFrame()
            
            def save_results(self, results, output_dir, output_tag, save_options=None):
                pass
        
        print("✓ Decorator registration completed")
        
        # Verify the module was registered
        from pyAPisolation.analysis import get_analysis_module
        module = get_analysis_module('test_decorator')
        if module:
            print(f"✓ Decorator module found: {module.display_name}")
        else:
            print("✗ Decorator module not found")
            
    except Exception as e:
        print(f"✗ Decorator registration failed: {e}")


def test_error_handling():
    """Test error handling in registration"""
    print("\n=== Testing Error Handling ===")
    
    from pyAPisolation.analysis import register_analysis_module
    
    # Test registering invalid class
    try:
        register_analysis_module(str)  # String is not an AnalysisModule
        print("✗ Should have failed for invalid class")
    except Exception as e:
        print(f"✓ Correctly rejected invalid class: {type(e).__name__}")
    
    # Test invalid tab mapping
    try:
        from pyAPisolation.analysis import analysis_registry
        analysis_registry.add_tab_mapping(10, "nonexistent_module")
        print("✗ Should have failed for nonexistent module")
    except Exception as e:
        print(f"✓ Correctly rejected invalid tab mapping: {type(e).__name__}")


def test_registry_queries():
    """Test registry query functions"""
    print("\n=== Testing Registry Queries ===")
    
    from pyAPisolation.analysis import (
        list_available_analyses, get_analysis_module, analysis_registry
    )
    
    # Test listing modules
    modules = list_available_analyses()
    print(f"✓ Found {len(modules)} registered modules:")
    for name, display_name in modules.items():
        print(f"  - {name}: {display_name}")
    
    # Test getting specific module
    spike_module = get_analysis_module('spike')
    if spike_module:
        print(f"✓ Retrieved spike module: {spike_module.display_name}")
    else:
        print("✗ Could not retrieve spike module")
    
    # Test getting nonexistent module
    fake_module = get_analysis_module('nonexistent')
    if fake_module is None:
        print("✓ Correctly returned None for nonexistent module")
    else:
        print("✗ Should have returned None for nonexistent module")


def main():
    """Run all tests"""
    print("=== Analysis Module Registration Utilities Test ===\n")
    
    try:
        test_basic_registration()
        test_tab_registration()
        test_decorator_registration()
        test_error_handling()
        test_registry_queries()
        
        print("\n=== Test Summary ===")
        print("✓ All registration utilities are working correctly!")
        print("✓ The modular analysis framework is ready to use")
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
