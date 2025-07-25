# pyAPisolation Modular Analysis Framework

The new modular analysis framework provides a clean, extensible architecture for electrophysiology data analysis while maintaining backward compatibility with existing code.

## Key Features

- **Modular Design**: Easy to add new analysis types
- **CLI Interface**: Command-line access for batch processing
- **Modern GUI**: Updated interface using the new framework
- **Backward Compatibility**: Legacy code continues to work
- **Parallel Processing**: Built-in support for multi-core analysis
- **Flexible Parameters**: Type-safe parameter handling
- **Progress Tracking**: Real-time progress updates

## Architecture Overview

```
pyAPisolation/
├── analysis/
│   ├── __init__.py          # Main module exports
│   ├── base.py              # Base classes and interfaces
│   ├── registry.py          # Analyzer registration system
│   ├── spike_analyzer.py    # Spike detection analysis
│   ├── subthreshold_analyzer.py  # Subthreshold analysis
│   ├── runner.py            # Batch processing runner
│   └── legacy.py            # Backward compatibility
├── cli.py                   # Command-line interface
├── gui/
│   └── modern_gui.py        # Updated GUI using new framework
└── examples/
    ├── usage_examples.py    # Programming examples
    └── analysis_configs.json # Example configurations
```

## Quick Start

### CLI Usage

```bash
# List available analyzers
python -m pyAPisolation.cli --list-analyzers

# Basic spike analysis
python -m pyAPisolation.cli spike /path/to/data/ --protocol "IC_STEPS"

# Subthreshold analysis with custom parameters
python -m pyAPisolation.cli subthreshold /path/to/data/ --time-after 75

# Batch processing with parallel execution
python -m pyAPisolation.cli spike "/path/**/*.abf" --parallel --output /results/
```

### Programmatic Usage

```python
from pyAPisolation.analysis import registry, AnalysisParameters, AnalysisRunner

# Single file analysis
parameters = AnalysisParameters(protocol_filter="IC_STEPS")
parameters.extra_params.update({'dv_cutoff': 7.0, 'min_height': 2.0})

analyzer = registry.get_analyzer('spike')
result = analyzer.analyze_file('path/to/file.abf', parameters)

# Batch analysis
runner = AnalysisRunner()
results = runner.run_batch(
    file_pattern='/path/to/data/*.abf',
    analyzer_name='spike',
    parameters=parameters,
    parallel=True
)
```

## Available Analyzers

### Spike Analyzer (`spike`)

Detects and analyzes action potentials using the IPFX framework.

**Key Parameters:**
- `dv_cutoff`: dV/dt threshold for spike detection (mV/s)
- `max_interval`: Max time from threshold to peak (seconds)
- `min_height`: Min threshold-to-peak height (mV)
- `min_peak`: Min peak voltage (mV)
- `thresh_frac`: Fraction of max dV/dt for threshold refinement

**Output:**
- Summary data: One row per file with spike counts and basic features
- Detailed data: Individual spike features for each sweep
- Sweep data: Running bin analysis of spike features

### Subthreshold Analyzer (`subthreshold`)

Analyzes passive membrane properties and subthreshold responses.

**Key Parameters:**
- `time_after`: Percentage of decay to analyze after current step
- `subt_sweeps`: Specific sweep numbers to analyze (None=auto)
- `start_sear`: Start time for analysis window (s)
- `end_sear`: End time for analysis window (s)

**Output:**
- Summary data: Averaged membrane properties per file
- Detailed data: Sweep-wise analysis results

## Configuration Files

You can use JSON configuration files to define analysis parameters:

```json
{
  "analyzer": "spike",
  "parameters": {
    "dv_cutoff": 7.0,
    "max_interval": 0.010,
    "min_height": 2.0,
    "min_peak": -10.0,
    "thresh_frac": 0.05
  },
  "output": {
    "format": "csv",
    "prefix": "spike_analysis_"
  }
}
```

## Creating Custom Analyzers

The framework is designed to be easily extensible:

```python
from pyAPisolation.analysis.base import BaseAnalyzer, AnalysisResult, AnalysisParameters

class MyCustomAnalyzer(BaseAnalyzer):
    @property
    def analysis_type(self) -> str:
        return "custom"
    
    def validate_parameters(self, parameters: AnalysisParameters) -> list:
        # Return list of validation errors
        return []
    
    def analyze_file(self, file_path: str, parameters: AnalysisParameters) -> AnalysisResult:
        # Implement your analysis logic
        result = AnalysisResult(
            analyzer_name=self.name,
            file_path=file_path,
            success=True
        )
        # ... analysis code ...
        return result

# Register your analyzer
from pyAPisolation.analysis import registry
registry.register('my_custom', MyCustomAnalyzer)
```

## GUI Integration

The modern GUI (`modern_gui.py`) demonstrates how to integrate the new framework with existing Qt interfaces:

```python
from pyAPisolation.gui.modern_gui import create_modern_gui

# Create enhanced GUI with new framework features
gui = create_modern_gui()
```

## Backward Compatibility

Legacy code continues to work through the compatibility layer:

```python
# Legacy function calls are automatically routed through the new framework
from pyAPisolation.featureExtractor import process_file
result = process_file('file.abf', param_dict, protocol_name)
```

## Migration Guide

### From Legacy GUI
1. The existing GUI continues to work unchanged
2. New features are available through `modern_gui.py`
3. Consider migrating custom analysis workflows to use the new analyzers

### From Legacy CLI Scripts
1. Update imports to use `pyAPisolation.cli`
2. Use the new parameter format (see examples)
3. Take advantage of built-in parallel processing

### From Legacy Batch Scripts
1. Replace custom batch processing with `AnalysisRunner`
2. Use the registry system for analyzer selection
3. Leverage built-in progress tracking and error handling

## Performance Considerations

- **Parallel Processing**: Enabled by default for batch analysis
- **Memory Management**: Large datasets are processed in chunks
- **Caching**: Analyzer instances are cached for reuse
- **Error Handling**: Robust error isolation prevents single file failures from stopping batch jobs

## Testing and Validation

The new framework includes comprehensive testing:
- Unit tests for all analyzers
- Integration tests for CLI and GUI
- Validation against legacy results
- Performance benchmarking

## Future Extensions

The modular design makes it easy to add:
- New analysis types (e.g., calcium imaging, field potentials)
- Advanced visualization components
- Machine learning integration
- Cloud processing capabilities
- Real-time analysis for online experiments

## Support and Documentation

- See `examples/usage_examples.py` for detailed code examples
- Use `--help` with CLI commands for parameter documentation
- Check `analysis_configs.json` for configuration templates
- Legacy documentation remains valid for existing features
