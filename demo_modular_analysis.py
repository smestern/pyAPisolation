#!/usr/bin/env python3
"""
Demo script showing the new modular analysis framework

This script demonstrates the key improvements and how to use the new system.
"""

def demo_registry_system():
    """Show how the registry system works"""
    print("=== Registry System Demo ===")
    
    from pyAPisolation.analysis import registry
    
    # List available analyzers
    analyzers = registry.list_analyzers()
    print(f"Available analyzers: {analyzers}")
    
    # Get analyzer information
    for name in analyzers:
        info = registry.get_analyzer_info(name)
        print(f"\n{name.upper()}:")
        print(f"  Type: {info['type']}")
        print(f"  Class: {info['class']}")
    
    return analyzers


def demo_parameter_system():
    """Show the new parameter system"""
    print("\n=== Parameter System Demo ===")
    
    from pyAPisolation.analysis import AnalysisParameters, registry
    
    # Create parameters
    params = AnalysisParameters(
        start_time=0.5,
        end_time=2.0,
        protocol_filter="IC_STEPS"
    )
    
    # Add analyzer-specific parameters
    params.extra_params.update({
        'dv_cutoff': 7.0,
        'min_height': 2.0,
        'thresh_frac': 0.05
    })
    
    print(f"Start time: {params.start_time}")
    print(f"Protocol filter: {params.protocol_filter}")
    print(f"DV cutoff: {params.get('dv_cutoff')}")
    print(f"Unknown param: {params.get('unknown', 'DEFAULT')}")
    
    # Show parameter schemas
    spike_analyzer = registry.get_analyzer('spike')
    schema = spike_analyzer.get_parameter_schema()
    print(f"\nSpike analyzer parameters:")
    for name, info in list(schema.items())[:3]:  # Show first 3
        print(f"  {name}: {info.get('description', 'No description')}")


def demo_modular_design():
    """Show how to create a simple custom analyzer"""
    print("\n=== Modular Design Demo ===")
    
    from pyAPisolation.analysis.base import BaseAnalyzer, AnalysisResult, AnalysisParameters
    from pyAPisolation.analysis import registry
    import pandas as pd
    import numpy as np
    
    class SimpleStatsAnalyzer(BaseAnalyzer):
        """Example custom analyzer that computes basic statistics"""
        
        @property
        def analysis_type(self) -> str:
            return "basic_statistics"
        
        def validate_parameters(self, parameters: AnalysisParameters) -> list:
            errors = []
            if parameters.start_time < 0:
                errors.append("start_time must be non-negative")
            return errors
        
        def analyze_file(self, file_path: str, parameters: AnalysisParameters) -> AnalysisResult:
            result = AnalysisResult(
                analyzer_name=self.name,
                file_path=file_path,
                success=False
            )
            
            try:
                # Simulate analysis (normally would load real file)
                print(f"  Analyzing {file_path}...")
                
                # Create fake results
                summary_data = pd.DataFrame({
                    'filename': [file_path],
                    'mean_voltage': [np.random.normal(-60, 5)],
                    'voltage_std': [np.random.uniform(1, 5)],
                    'analysis_type': ['basic_stats']
                })
                
                result.summary_data = summary_data
                result.metadata['n_sweeps'] = np.random.randint(5, 20)
                result.success = True
                
            except Exception as e:
                result.add_error(f"Analysis failed: {str(e)}")
            
            return result
    
    # Register the custom analyzer
    registry.register('simple_stats', SimpleStatsAnalyzer)
    
    print(f"Registered custom analyzer!")
    print(f"Available analyzers now: {registry.list_analyzers()}")
    
    # Test the custom analyzer
    custom_analyzer = registry.get_analyzer('simple_stats')
    params = AnalysisParameters(start_time=0.0)
    
    result = custom_analyzer.analyze_file('demo_file.abf', params)
    if result.success:
        print(f"Custom analysis successful!")
        print(f"Result: {result.summary_data.iloc[0]['mean_voltage']:.1f} mV")
    else:
        print(f"Custom analysis failed: {result.errors}")


def demo_cli_interface():
    """Show CLI examples"""
    print("\n=== CLI Interface Demo ===")
    
    cli_examples = [
        "# List available analyzers",
        "python -m pyAPisolation.cli --list-analyzers",
        "",
        "# Run spike analysis on a directory", 
        "python -m pyAPisolation.cli spike /path/to/data/ --protocol 'IC_STEPS'",
        "",
        "# Subthreshold analysis with custom parameters",
        "python -m pyAPisolation.cli subthreshold /path/to/data/ \\",
        "    --time-after 75 --output /results/ --format excel",
        "",
        "# Parallel batch processing",
        "python -m pyAPisolation.cli spike '/path/**/*.abf' \\",
        "    --parallel --workers 4 --verbose",
        "",
        "# Using a configuration file",
        "python -m pyAPisolation.cli spike /path/to/data/ \\",
        "    --config examples/analysis_configs.json"
    ]
    
    for line in cli_examples:
        print(line)


def demo_backward_compatibility():
    """Show how legacy code still works"""
    print("\n=== Backward Compatibility Demo ===")
    
    print("Legacy imports still work:")
    print("  from pyAPisolation.featureExtractor import process_file")
    print("  # This now uses the new spike analyzer under the hood")
    
    print("\nLegacy parameter dictionaries are automatically converted:")
    legacy_params = {
        'dv_cutoff': 7.0,
        'start': 0.0,
        'end': 2.0,
        'min_height': 2.0
    }
    print(f"  Legacy params: {legacy_params}")
    
    # Show conversion (without actually running analysis)
    from pyAPisolation.analysis.legacy import LegacyAnalysisWrapper
    wrapper = LegacyAnalysisWrapper()
    converted = wrapper._convert_legacy_spike_params(legacy_params)
    print(f"  Converted start_time: {converted.start_time}")
    print(f"  Converted dv_cutoff: {converted.get('dv_cutoff')}")


def demo_benefits():
    """Highlight the key benefits of the new system"""
    print("\n=== Key Benefits ===")
    
    benefits = [
        "✓ Modular design - easy to add new analysis types",
        "✓ Type-safe parameters with validation",
        "✓ Built-in CLI with argument parsing",
        "✓ Parallel processing for batch analysis", 
        "✓ Progress tracking and error handling",
        "✓ Flexible output formats (CSV, Excel)",
        "✓ Configuration file support",
        "✓ Complete backward compatibility",
        "✓ Extensible registry system",
        "✓ Modern GUI integration ready"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")


def main():
    """Run all demos"""
    print("pyAPisolation Modular Analysis Framework Demo")
    print("=" * 50)
    
    try:
        # Core system demos
        demo_registry_system()
        demo_parameter_system() 
        demo_modular_design()
        
        # Usage demos
        demo_cli_interface()
        demo_backward_compatibility()
        
        # Summary
        demo_benefits()
        
        print("\n" + "=" * 50)
        print("Demo complete! The new framework is ready to use.")
        print("See MODULAR_ANALYSIS_README.md for detailed documentation.")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the correct directory.")
    except Exception as e:
        print(f"Demo error: {e}")


if __name__ == "__main__":
    main()
