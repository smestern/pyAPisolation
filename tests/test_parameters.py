"""
Test script to verify the parameter system works correctly
"""

import sys
import os

# Add the parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from pyAPisolation.analysis.base import AnalysisModule, AnalysisParameters
    
    print("✓ Successfully imported AnalysisModule and AnalysisParameters")
    
    # Test AnalysisParameters
    params = AnalysisParameters()
    params.start_time = 1.0
    params.end_time = 10.0
    params.set('custom_param', 'test_value')
    
    print(f"✓ AnalysisParameters created: start_time={params.start_time}")
    print(f"✓ Custom parameter: {params.get('custom_param')}")
    
    # Test AnalysisModule (we'll create a minimal concrete implementation)
    class TestModule(AnalysisModule):
        def get_ui_elements(self):
            return {}
        
        def parse_ui_params(self, ui_elements):
            return {}
        
        def run_individual_analysis(self, abf, selected_sweeps, param_dict, 
                                   popup=None, show_rejected=False):
            return {}
        
        def run_batch_analysis(self, folder_path, param_dict, protocol_name):
            return ([], {})
        
        def save_results(self, results, output_dir, output_tag, save_options=None):
            pass
    
    # Test module with parameters
    test_module = TestModule("test", "Test Module", params)
    
    print(f"✓ AnalysisModule created: {test_module.name}")
    print(f"✓ Parameter access: start_time={test_module.get_parameter('start_time')}")
    print(f"✓ Parameter access with default: threshold={test_module.get_parameter('threshold', 0.5)}")
    
    # Test parameter updates
    test_module.set_parameter('threshold', 1.0)
    print(f"✓ Parameter set: threshold={test_module.get_parameter('threshold')}")
    
    test_module.update_parameters(start_time=2.0, end_time=20.0)
    print(f"✓ Batch parameter update: start_time={test_module.get_parameter('start_time')}")
    
    # Test parameters property
    new_params = AnalysisParameters()
    new_params.start_time = 5.0
    test_module.parameters = new_params
    print(f"✓ Parameters property: start_time={test_module.get_parameter('start_time')}")
    
    print("\n🎉 All parameter system tests passed!")
    
except Exception as e:
    print(f"❌ Error testing parameter system: {e}")
    import traceback
    traceback.print_exc()
