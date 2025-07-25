"""
Example usage of the new modular analysis framework

This script demonstrates how to use the new analysis system
for both programmatic access and as a guide for CLI usage.
"""

import os
import sys
from pathlib import Path

# Add the package to the path for examples
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyAPisolation.analysis import (
    registry, AnalysisParameters, AnalysisRunner, 
    SpikeAnalyzer, SubthresholdAnalyzer
)


def example_single_file_analysis():
    """Example: Analyze a single file"""
    print("=== Single File Analysis Example ===")
    
    # Example file path (adjust to your data)
    file_path = "path/to/your/data.abf"
    
    # Create parameters for spike analysis
    parameters = AnalysisParameters(
        start_time=0.0,
        end_time=0.0,  # 0 means use full recording
        protocol_filter="IC_STEPS"
    )
    
    # Set spike-specific parameters
    parameters.extra_params.update({
        'dv_cutoff': 7.0,
        'max_interval': 0.010,
        'min_height': 2.0,
        'min_peak': -10.0,
        'thresh_frac': 0.05
    })
    
    # Get the analyzer and run analysis
    spike_analyzer = registry.get_analyzer('spike')
    result = spike_analyzer.analyze_file(file_path, parameters)
    
    if result.success:
        print(f"Analysis successful!")
        print(f"Summary data shape: {result.summary_data.shape}")
        print(f"Detailed data shape: {result.detailed_data.shape}")
    else:
        print(f"Analysis failed: {result.errors}")


def example_batch_analysis():
    """Example: Batch analysis with progress tracking"""
    print("\n=== Batch Analysis Example ===")
    
    # Example data directory
    data_dir = "path/to/your/data/"
    
    # Create parameters
    parameters = AnalysisParameters(
        start_time=0.0,
        end_time=0.0,
        protocol_filter="CC_STEPS"
    )
    
    # Subthreshold-specific parameters
    parameters.extra_params.update({
        'time_after': 50.0,
        'savfilter': 0
    })
    
    # Create runner
    runner = AnalysisRunner(registry)
    
    # Progress callback
    def show_progress(current, total):
        percent = int(100 * current / total)
        print(f"Progress: {percent}% ({current}/{total})")
    
    # Run batch analysis
    results = runner.run_batch(
        file_pattern=os.path.join(data_dir, "*.abf"),
        analyzer_name='subthreshold',
        parameters=parameters,
        parallel=True,
        progress_callback=show_progress
    )
    
    # Print summary
    stats = runner.get_summary_stats()
    print(f"Batch analysis complete:")
    print(f"  Total files: {stats['total_files']}")
    print(f"  Successful: {stats['successful']}")
    print(f"  Failed: {stats['failed']}")
    
    # Save results
    if results:
        output_dir = "results/"
        saved_files = runner.save_results(
            output_dir=output_dir,
            file_prefix="batch_analysis_",
            save_format="csv"
        )
        print(f"Results saved to: {saved_files}")


def example_single_sweep_analysis():
    """Example: Analyze individual sweeps"""
    print("\n=== Single Sweep Analysis Example ===")
    
    import numpy as np
    
    # Example sweep data (replace with real data)
    time_data = np.linspace(0, 2.0, 20000)  # 2 seconds at 10kHz
    voltage_data = np.random.randn(20000) * 2 - 60  # Random voltage around -60mV
    current_data = np.zeros(20000)  # No current injection
    
    sweep_data = {
        'time': time_data,
        'voltage': voltage_data,
        'current': current_data
    }
    
    # Create parameters
    parameters = AnalysisParameters()
    parameters.extra_params.update({
        'dv_cutoff': 7.0,
        'max_interval': 0.010,
        'min_height': 2.0
    })
    
    # Analyze single sweep
    spike_analyzer = registry.get_analyzer('spike')
    
    try:
        features = spike_analyzer.analyze_sweep(sweep_data, parameters)
        print(f"Sweep analysis result: {features}")
    except NotImplementedError as e:
        print(f"Single sweep analysis not implemented: {e}")


def example_custom_analyzer():
    """Example: Create and register a custom analyzer"""
    print("\n=== Custom Analyzer Example ===")
    
    from pyAPisolation.analysis.base import BaseAnalyzer, AnalysisResult, AnalysisParameters
    import pandas as pd
    
    class BasicStatsAnalyzer(BaseAnalyzer):
        """Simple analyzer that computes basic voltage statistics"""
        
        @property
        def analysis_type(self) -> str:
            return "basic_stats"
        
        def validate_parameters(self, parameters: AnalysisParameters) -> list:
            return []  # No specific validation needed
        
        def analyze_file(self, file_path: str, parameters: AnalysisParameters) -> AnalysisResult:
            result = AnalysisResult(
                analyzer_name=self.name,
                file_path=file_path,
                success=False
            )
            
            try:
                import pyabf
                abf = pyabf.ABF(file_path)
                
                stats_data = []
                for sweep in abf.sweepList:
                    abf.setSweep(sweep)
                    stats_data.append({
                        'sweep': sweep,
                        'mean_voltage': np.mean(abf.sweepY),
                        'std_voltage': np.std(abf.sweepY),
                        'min_voltage': np.min(abf.sweepY),
                        'max_voltage': np.max(abf.sweepY),
                        'filename': abf.abfID
                    })
                
                result.summary_data = pd.DataFrame(stats_data)
                result.success = True
                
            except Exception as e:
                result.add_error(str(e))
            
            return result
    
    # Register the custom analyzer
    registry.register('basic_stats', BasicStatsAnalyzer)
    
    print(f"Registered custom analyzer. Available analyzers: {registry.list_analyzers()}")
    
    # Use the custom analyzer
    # analyzer = registry.get_analyzer('basic_stats')
    # result = analyzer.analyze_file("path/to/file.abf", AnalysisParameters())


def example_parameter_schemas():
    """Example: Explore parameter schemas"""
    print("\n=== Parameter Schemas Example ===")
    
    for analyzer_name in registry.list_analyzers():
        analyzer = registry.get_analyzer(analyzer_name)
        schema = analyzer.get_parameter_schema()
        
        print(f"\n{analyzer_name.upper()} ANALYZER PARAMETERS:")
        for param_name, param_info in schema.items():
            print(f"  {param_name}:")
            print(f"    Type: {param_info.get('type', 'unknown')}")
            print(f"    Default: {param_info.get('default', 'N/A')}")
            print(f"    Description: {param_info.get('description', 'N/A')}")


def cli_examples():
    """Print CLI usage examples"""
    print("\n=== CLI Usage Examples ===")
    
    examples = [
        "# List available analyzers",
        "python -m pyAPisolation.cli --list-analyzers",
        "",
        "# Basic spike analysis on a directory",
        "python -m pyAPisolation.cli spike /path/to/data/ --protocol IC_STEPS",
        "",
        "# Spike analysis with custom parameters",
        "python -m pyAPisolation.cli spike /path/to/data/ --dv-cutoff 5.0 --min-height 1.5",
        "",
        "# Subthreshold analysis with output to Excel",
        "python -m pyAPisolation.cli subthreshold /path/to/data/ --format excel --time-after 75",
        "",
        "# Batch analysis with parallel processing",
        "python -m pyAPisolation.cli spike '/path/**/*.abf' --parallel --output /results/",
        "",
        "# Using a configuration file",
        "python -m pyAPisolation.cli spike /path/to/data/ --config analysis_config.json",
        "",
        "# Verbose output with progress tracking",
        "python -m pyAPisolation.cli subthreshold /path/to/data/ --verbose"
    ]
    
    for example in examples:
        print(example)


if __name__ == "__main__":
    print("pyAPisolation Modular Analysis Framework Examples")
    print("=" * 60)
    
    # Show available analyzers
    print(f"Available analyzers: {registry.list_analyzers()}")
    
    # Run examples (comment out file-dependent ones for demo)
    # example_single_file_analysis()
    # example_batch_analysis()
    example_single_sweep_analysis()
    example_custom_analyzer()
    example_parameter_schemas()
    cli_examples()
    
    print("\n" + "=" * 60)
    print("For more information, see the documentation or use --help with the CLI")
