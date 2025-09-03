# pyAPisolation Development Guide for AI Coding Agents

## Architecture Overview

pyAPisolation is a **dual-paradigm** electrophysiology analysis package supporting both:
- **Legacy GUI-driven workflows** (PySide2-based, `/gui/spikeFinder.py`)  
- **Modern modular CLI/programmatic analysis** (`/analysis/` framework)

The package processes **ABF files** (Axon Binary Format) for patch-clamp data analysis with three main interaction modes: GUI applications, CLI tools, and programmatic APIs.

### Core Data Flow

```
ABF Files → cellData objects → Analysis Modules → Results (DataFrames/CSVs)
          ↘ GUI (PySide2) ↗
          ↘ CLI (argparse) ↗  
```

## Critical Architectural Components

### 1. Analysis Framework (`pyAPisolation/analysis/`)
The **modular analysis system** is the preferred approach:
- `base.py`: `AnalysisModule` abstract base + `AnalysisResult`/`AnalysisParameters` data classes
- `registry.py`: Plugin-style module registration (`registry.register_module()`)  
- `builtin_modules.py`: `SpikeAnalysisModule`, `SubthresholdAnalysisModule`, `ResistanceLadder`
- `runner.py`: `AnalysisRunner` for batch processing with multiprocessing support

**Key Pattern**: Modules inherit from `AnalysisModule` and implement `analyze()` and `run_batch_analysis()`

### 2. Data Layer (`dataset.py`)
- `cellData` class: Main data container following pyABF conventions
- Properties: `dataX` (time), `dataY` (response), `dataC` (command)
- Supports both file loading and direct array initialization
- Protocol-aware via `protocolList` parameter

### 3. GUI System (`pyAPisolation/gui/`)
- **Main GUI**: `spikeFinder.py` (MDI interface, PySide2 + pyqtgraph)
- **Database Builder**: `databaseBuilder.py` (Excel-like cell/protocol management)
- **Modern GUI**: `modern_gui.py` (newer streamlined interface)

**Critical**: GUI uses **hybrid approach** - tries modular analysis first, falls back to legacy:
```python
def run_analysis(self):
    current_module = self.get_current_analysis_module()
    if current_module is None:
        return self._run_analysis_legacy()
    try:
        return self._run_analysis_modular(current_module)
    except NotImplementedError:
        return self._run_analysis_legacy()
```

## Essential Development Patterns

### Adding New Analysis Modules
1. **Inherit from `AnalysisModule`** in `analysis/base.py`
2. **Implement required methods**: `get_default_parameters()`, `analyze()`, `run_batch_analysis()`
3. **Register with framework**: `registry.register_module(YourModule())`
4. **Handle UI integration**: Implement `parse_ui_params()` for GUI compatibility

### Data Processing Conventions
- **Always use `cellData` objects** for consistent data handling
- **Preserve sweep structure** - many analyses operate per-sweep then aggregate
- **Handle missing data gracefully** - ABF files can have incomplete protocols
- **Use ipfx integration** where possible (`ipfx.feature_extractor`, `ipfx.sweep`)

### GUI Integration Points
- **Parameter binding**: UI elements → `get_analysis_params_modular()` → module parameters
- **Progress tracking**: Use `QProgressDialog` with `popup.setValue()` in analysis loops
- **Result display**: Convert analysis outputs to pandas DataFrames for table views
- **File management**: GUI handles file selection, passes paths to analysis modules

## Key Commands & Workflows

### Entry Points (from `pyproject.toml`)
```bash
# CLI entry point
spike_finder data_folder --analyzer spike --output results/

# GUI applications  
python pyAPisolation/bin/run_APisolation_gui.py        # Main analysis GUI
python pyAPisolation/bin/run_modern_gui.py            # Modern interface
python pyAPisolation/bin/run_builddatabase.py         # Database builder
```

### Development Commands
```bash
# Install with GUI dependencies
pip install -e ".[gui]"

# Run tests 
python -m pytest tests/

# Build standalone executables (PyInstaller specs in /bin/)
python pyAPisolation/bin/run_APisolation_gui.spec
```

### Analysis Module Testing Pattern
```python
from pyAPisolation.analysis import registry
from pyAPisolation.dataset import cellData

# Get module and run analysis
module = registry.get_analyzer('spike')
data = cellData('path/to/file.abf')
results = module.analyze(data, selected_sweeps=[0,1,2], param_dict={})
```

## Integration Dependencies

### Required External Packages
- **pyABF**: ABF file reading (Axon Instruments format)
- **ipfx**: Allen Institute feature extraction (`feature_extractor`, `sweep`)
- **PySide2**: GUI framework (NOT PyQt5 - licensing reasons)  
- **pyqtgraph**: High-performance plotting

### Data Format Constraints  
- **ABF files** must contain proper sweep/protocol metadata
- **Clamp modes** affect analysis: Current Clamp vs Voltage Clamp detection
- **Sampling rates** vary - analysis modules must handle different `dt` values
- **Protocol names** used for file organization and analysis selection

## Critical Development Notes

### Backward Compatibility Requirements
- **Legacy GUI code** in `gui/spikeFinder.py` must remain functional
- **File formats**: CSV outputs maintain column compatibility for existing workflows  
- **Parameter names**: Changing analysis parameter names breaks saved configurations

### Performance Considerations
- **Multiprocessing**: Analysis modules support `n_jobs` parameter for parallel processing
- **Memory management**: Large ABF files (>1GB) require streaming approaches
- **GUI responsiveness**: Long analyses use `QProgressDialog` and `QApplication.processEvents()`

### Windows-Specific Patterns
- **File paths**: Use `os.path.join()` for cross-platform compatibility
- **PyInstaller**: `.spec` files in `/bin/` for standalone executable generation
- **PySide2 deployment**: Includes platform-specific DLLs (`qwindows.dll`, etc.)

When modifying this codebase:
1. **Test both GUI and CLI workflows** - they share analysis modules but have different parameter handling
2. **Verify with real ABF files** - synthetic data often misses edge cases
3. **Check multiprocessing compatibility** - avoid global state in analysis modules  
4. **Maintain parameter backward compatibility** - GUI configurations are saved/loaded
