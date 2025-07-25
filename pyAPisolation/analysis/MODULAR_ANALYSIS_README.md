# Modular Analysis Framework for pyAPisolation GUI

## Overview

The pyAPisolation GUI has been refactored to support a modular analysis framework while preserving all legacy functionality. This allows you to easily add new analysis types without modifying the core GUI code.

## Key Features

- **Backward Compatibility**: All existing spike and subthreshold analysis functionality is preserved
- **Extensible**: Easy to add new analysis types
- **Automatic Fallback**: If modular analysis fails, the system falls back to legacy implementation
- **Clean Separation**: Analysis logic is separated from GUI logic

## Architecture

### Core Components

1. **`pyAPisolation.analysis.AnalysisModule`**: Abstract base class for all analysis types
2. **`pyAPisolation.analysis.AnalysisRegistry`**: Manages and provides access to analysis modules
3. **Built-in Modules**: `SpikeAnalysisModule` and `SubthresholdAnalysisModule` wrap existing functionality
4. **Utility Functions**: Easy registration and management functions
5. **Updated GUI**: `analysis_gui` class now supports both modular and legacy analysis

### Module Structure

```
pyAPisolation/
├── analysis/                    # New analysis framework
│   ├── __init__.py             # Main package interface
│   ├── base.py                 # Abstract AnalysisModule class
│   ├── registry.py             # AnalysisRegistry class
│   ├── builtin_modules.py      # Legacy spike & subthreshold modules
│   └── utilities.py            # Registration utilities & decorators
└── gui/
    └── spikeFinder.py          # Updated to use new framework
```

### Class Hierarchy

```
AnalysisModule (ABC)
├── SpikeAnalysisModule (legacy spike analysis)
├── SubthresholdAnalysisModule (legacy subthreshold analysis)
└── YourCustomAnalysisModule (your new analysis)
```

## Adding a New Analysis Module

### Step 1: Create Your Analysis Module

Create a new class that inherits from `AnalysisModule`:

```python
from pyAPisolation.analysis import AnalysisModule

class MyCustomAnalysisModule(AnalysisModule):
    def __init__(self):
        super().__init__("my_analysis", "My Custom Analysis")
    
    def get_ui_elements(self):
        """Define what UI controls your analysis needs"""
        return {
            'start_time': 'float',
            'end_time': 'float',
            'my_parameter': 'float',
            'enable_option': 'bool'
        }
    
    def parse_ui_params(self, ui_elements):
        """Extract parameters from UI elements"""
        return {
            'start': float(ui_elements['start_time'].text()),
            'end': float(ui_elements['end_time'].text()),
            'parameter': float(ui_elements['my_parameter'].text()),
            'option': ui_elements['enable_option'].isChecked()
        }
    
    def run_individual_analysis(self, abf, selected_sweeps, param_dict, popup=None, show_rejected=False):
        """Run analysis on a single file for preview"""
        # Your analysis logic here
        results = {}
        # ... implement your analysis ...
        return {
            'my_results': results,
            'spike_df': None,  # Set if your analysis produces spike data
            'rejected_spikes': None,
            'subthres_df': None
        }
    
    def run_batch_analysis(self, folder_path, param_dict, protocol_name):
        """Run analysis on multiple files"""
        # Your batch processing logic here
        return results_dataframe
    
    def save_results(self, results, output_dir, output_tag, save_options=None):
        """Save your analysis results"""
        # Your save logic here
        pass
```

### Step 2: Register Your Module

You have several options for registering your module:

#### Option A: Using Utility Functions (Recommended)

```python
# Simple registration
from pyAPisolation.analysis import register_analysis_module
my_module = register_analysis_module(MyCustomAnalysisModule)

# Register with specific tab
from pyAPisolation.analysis import register_analysis_with_tab
my_module = register_analysis_with_tab(MyCustomAnalysisModule, 2)  # Tab index 2
```

#### Option B: Using Decorator (Automatic Registration)

```python
from pyAPisolation.analysis import analysis_module

@analysis_module(name="my_analysis", display_name="My Custom Analysis", tab_index=2)
class MyCustomAnalysisModule(AnalysisModule):
    # ... your implementation ...
```

#### Option C: Manual Registration

```python
from pyAPisolation.analysis import analysis_registry
my_module = MyCustomAnalysisModule()
analysis_registry.register_module(my_module)

# Optionally add tab mapping
analysis_registry.add_tab_mapping(2, "my_analysis")
```

### Step 3: Query and Use Your Module

```python
# List all available modules
from pyAPisolation.analysis import list_available_analyses
print(list_available_analyses())

# Get a specific module
from pyAPisolation.analysis import get_analysis_module
my_module = get_analysis_module("my_analysis")
```

## Registration Utility Functions

The framework provides several utility functions to make module registration easier:

### Core Utilities

- **`register_analysis_module(module_class, *args, **kwargs)`**: Register a module class
- **`register_analysis_with_tab(module_class, tab_index, *args, **kwargs)`**: Register with tab mapping
- **`list_available_analyses()`**: Get dictionary of all registered modules
- **`get_analysis_module(name)`**: Retrieve a specific module by name

### Decorator

- **`@analysis_module(name=None, display_name=None, tab_index=None)`**: Automatic registration decorator

### Registry Methods

The `AnalysisRegistry` class provides additional methods:

- **`register_module(module)`**: Register a module instance
- **`unregister_module(name)`**: Remove a module
- **`get_module_by_tab(tab_index)`**: Get module by tab index
- **`list_modules_detailed()`**: Get detailed module information
- **`add_tab_mapping(tab_index, module_name)`**: Add tab mapping

### Step 3: Add UI Controls (if needed)

If your analysis needs new UI controls, you can either:

1. **Use existing controls**: Map to existing UI elements that have similar data types
2. **Add new controls**: Modify the `.ui` file to add new controls and update the `bind_ui()` method

## Required Methods

Every analysis module must implement these abstract methods:

### `get_ui_elements()`
Returns a dictionary mapping UI element names to their types:
```python
{
    'element_name': 'type',  # 'float', 'int', 'bool', 'string', 'combo'
    'start_time': 'float',
    'enable_filter': 'bool'
}
```

### `parse_ui_params(ui_elements)`
Converts UI elements to analysis parameters:
```python
def parse_ui_params(self, ui_elements):
    return {
        'start': float(ui_elements['start_time'].text()),
        'filter': ui_elements['enable_filter'].isChecked()
    }
```

### `run_individual_analysis(...)`
Runs analysis on a single file for preview:
```python
def run_individual_analysis(self, abf, selected_sweeps, param_dict, popup=None, show_rejected=False):
    # Return dict with your results
    return {'my_data': dataframe}
```

### `run_batch_analysis(...)`
Runs analysis on multiple files:
```python
def run_batch_analysis(self, folder_path, param_dict, protocol_name):
    # Return results (dataframe or tuple of dataframes)
    return results
```

### `save_results(...)`
Saves analysis results:
```python
def save_results(self, results, output_dir, output_tag, save_options=None):
    # Save your results to files
    pass
```

## Optional Methods

### `get_plot_data(results, sweep_number=None)`
Extract data for custom plotting:
```python
def get_plot_data(self, results, sweep_number=None):
    # Return dict with plot data or None for default plotting
    return plot_data
```

## UI Element Types

The framework supports these UI element types:

- **`'float'`**: Numeric input (LineEdit with float parsing)
- **`'int'`**: Integer input (LineEdit with int parsing)
- **`'bool'`**: Checkbox (CheckBox.isChecked())
- **`'string'`**: Text input (LineEdit.text())
- **`'combo'`**: Dropdown selection (ComboBox.currentText())

## Legacy Compatibility

The framework maintains full backward compatibility:

- All existing analysis methods work unchanged
- If a modular analysis fails, it automatically falls back to legacy implementation
- Existing UI controls and workflows are preserved
- No changes needed to existing analysis code

## Migration Strategy

You can migrate existing analysis gradually:

1. **Phase 1**: Use framework with legacy fallback (current state)
2. **Phase 2**: Implement new analyses using modular framework
3. **Phase 3**: Optionally migrate legacy analyses to modular framework
4. **Phase 4**: Remove legacy fallback code (optional)

## Examples

See `example_custom_analysis.py` for complete working examples:

- **`ExampleCustomAnalysisModule`**: Basic statistics analysis
- **`AdvancedCustomAnalysisModule`**: Advanced example with multiple UI types

## Benefits

1. **Modularity**: Each analysis is self-contained
2. **Testability**: Analysis modules can be tested independently
3. **Reusability**: Modules can be used outside the GUI
4. **Maintainability**: Clear separation of concerns
5. **Extensibility**: Easy to add new features without touching core code

## Error Handling

The framework includes robust error handling:

- Automatic fallback to legacy implementation
- Graceful handling of missing UI elements
- Informative error messages for debugging

## Best Practices

1. **Start Simple**: Begin with basic functionality and add complexity gradually
2. **Handle Missing UI**: Always provide defaults for optional UI elements
3. **Document Parameters**: Clearly document what each parameter does
4. **Test Thoroughly**: Test both individual and batch analysis modes
5. **Error Handling**: Include try/catch blocks for robust operation
6. **Follow Naming**: Use descriptive names for modules and parameters

## Future Enhancements

The framework is designed to support future enhancements:

- Dynamic UI generation based on module requirements
- Plugin system for external modules
- Configuration file support
- Analysis pipelines (chaining multiple analyses)
- Parallel processing support
- Real-time analysis capabilities
