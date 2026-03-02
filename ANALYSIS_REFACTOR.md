# pyAPisolation Analysis Framework Refactor — Changelog & Migration Guide

## Overview

The `pyAPisolation.analysis` submodule was rebuilt from scratch to be **simple, modular, and user-friendly**. The old framework (`AnalysisModule`, `AnalysisParameters`, `Parameter`, `AnalysisRegistry`, `AnalysisRunner`) has been replaced with a single base class, a flat function registry, and a standalone batch runner.

**Design goals:**
- A novice programmer writes one class with one method (`analyze`) and it works
- Parameters are just typed class attributes — no `Parameter` objects, no `AnalysisParameters` dataclass
- The framework handles input normalization, sweep iteration, batching, and result packaging
- Legacy `featureExtractor.py` is untouched and fully importable

---

## Files Changed

### New Files Created

| File | Purpose |
|------|---------|
| `pyAPisolation/analysis/result.py` | `AnalysisResult` dataclass — lightweight result container with `to_dataframe()` and `concatenate()` |
| `pyAPisolation/analysis/base.py` | `AnalysisBase` — the single base class; handles input normalization, sweep iteration, parameter introspection |
| `pyAPisolation/analysis/registry.py` | `register()`, `get()`, `list_modules()`, `get_all()`, `clear()` — flat module registry |
| `pyAPisolation/analysis/runner.py` | `run_batch()`, `save_results()` — batch processing with optional `ProcessPoolExecutor` |
| `pyAPisolation/analysis/builtins/__init__.py` | Auto-registers built-in modules on import |
| `pyAPisolation/analysis/builtins/spike.py` | `SpikeAnalysis` — wraps `featureExtractor.analyze_sweep()` |
| `pyAPisolation/analysis/builtins/subthreshold.py` | `SubthresholdAnalysis` — wraps `patch_subthres.subthres_a()` |
| `pyAPisolation/analysis/builtins/example.py` | `PeakDetector` — minimal example for new users (~10 lines of user code) |
| `pyAPisolation/analysis/__init__.py` | Public API exports + backward-compatible shims for GUI |
| `tests/test_analysis_framework.py` | 24 tests covering results, base class, registry, demo data integration, and legacy imports |

### Old Files Preserved (Renamed)

| Original | Renamed To |
|----------|------------|
| `analysis/base.py` | `analysis/_legacy_base.py` |
| `analysis/builtin_modules.py` | `analysis/_legacy_builtin_modules.py` |
| `analysis/registry.py` | `analysis/_legacy_registry.py` |
| `analysis/runner.py` | `analysis/_legacy_runner.py` |
| `analysis/utilities.py` | `analysis/_legacy_utilities.py` |
| `analysis/__init__.py` | `analysis/_legacy___init__.py` |

### Bug Fixes (Unrelated to Refactor)

| File | Fix |
|------|-----|
| `pyAPisolation/patch_subthres.py` line 132 | Removed `-> tuple[float, ...]` return annotation (Python 3.8 incompatible) from `exp_decay_factor()` |
| `pyAPisolation/patch_subthres.py` line 264 | Removed `-> tuple[float, float]` return annotation from `compute_sag()` |

---

## How to Write a Custom Analysis Module

```python
from pyAPisolation.analysis import AnalysisBase, register
import numpy as np

class MyAnalysis(AnalysisBase):
    """Describe what your analysis does."""
    
    name = "my_analysis"                  # Unique ID for registry lookup
    display_name = "My Custom Analysis"   # Human-readable label
    sweep_mode = "per_sweep"              # "per_sweep" or "per_file"

    # Parameters — just typed class attributes with defaults
    threshold: float = -20.0
    window_ms: int = 5

    def analyze(self, x, y, c, **kwargs):
        """
        x: time array (1-D for per_sweep, 2-D for per_file)
        y: voltage/response array
        c: current/command array
        kwargs: sweep_number, file_path, celldata (injected by framework)
        
        Returns: a flat dict of results
        """
        peak_v = float(np.max(y))
        peak_t = float(x[np.argmax(y)])
        return {
            "peak_voltage": peak_v,
            "peak_time": peak_t,
            "above_threshold": peak_v > self.threshold,
        }

# Register it
register(MyAnalysis)
```

### Running Your Module

```python
from pyAPisolation.analysis import get, run_batch, save_results

# Single file
module = get("my_analysis")
result = module.run(file="path/to/recording.abf")
print(result.to_dataframe())

# Batch processing
result = run_batch(module, "data_folder/", protocol_filter="IC1", n_jobs=4)
save_results(result, "output/", tag="experiment1")

# From raw arrays
result = module.run(x=time_array, y=voltage_array, c=current_array)

# Override parameters at runtime
module.set_parameters(threshold=-30.0)
result = module.run(file="recording.abf")
```

---

## Old API → New API Migration

| Old API | New API | Notes |
|---------|---------|-------|
| `class Foo(AnalysisModule):` | `class Foo(AnalysisBase):` | |
| `super().__init__("name", "display")` | `name = "name"` as class attr | No positional `__init__` args |
| `Parameter("x", float, 7.0, ...)` | `x: float = 7.0` | Just a class attribute |
| `AnalysisParameters(start_time=..., ...)` | `module.set_parameters(start=...)` | No wrapper class needed |
| `module.get_ui_elements()` | `module.get_parameters()` | Returns `{name: {type, default, value}}` |
| `module.parse_ui_params(widgets)` | `module.set_parameters(**dict)` | Framework handles type coercion |
| `module.run_individual_analysis(abf, sweeps, params, popup)` | `module.run(celldata=data, selected_sweeps=sweeps)` | Returns `AnalysisResult` |
| `module.run_batch_analysis(folder, params, protocol)` | `run_batch(module, folder, protocol_filter=protocol)` | Standalone function |
| `module.save_results(results, dir, tag, opts)` | `save_results(result, dir, tag, fmt)` | Standalone function |
| `module.analyze(data, sweeps, param_dict)` | `module.analyze(x, y, c, **kwargs)` | Receives arrays, not ABF objects |
| `AnalysisRunner(registry).run_batch(...)` | `run_batch(module, ...)` | No runner class |
| `registry.get_module(name)` | `get(name)` | Or `registry.get_module(name)` via shim |
| `registry.register_module(module)` | `register(module)` | Or `registry.register_module(m)` via shim |
| `result.summary_data` / `result.detailed_data` | `result.to_dataframe()` | Single unified DataFrame |
| `result.analyzer_name` | `result.name` | Field renamed |
| `result.sweep_data` | `result.sweep_results` | List of dicts |

---

## Proposed GUI Updates

The GUI files (`spikeFinder.py`, `modern_gui.py`, `example_custom_analysis.py`) currently use backward-compatibility shims that prevent crashes but effectively always fall back to legacy code paths. Below are the proposed changes to fully integrate the new framework.

### Priority 1: `spikeFinder.py` (Main GUI)

The spikeFinder currently works because every call to the modular API throws `AttributeError`, which is caught and falls back to legacy. To actually use the new framework:

#### 1a. Replace `get_analysis_params_modular()`

**Current** (lines ~617–641):
```python
def get_analysis_params_modular(self):
    module = self.get_current_analysis_module()
    ui_elements = module.get_ui_elements()      # OLD API
    return module.parse_ui_params(ui_elements)   # OLD API
```

**Proposed**:
```python
def get_analysis_params_modular(self):
    module = self.get_current_analysis_module()
    if module is None:
        return self.get_analysis_params()  # legacy fallback
    
    # Read widget values and push them into the module
    params = module.get_parameters()
    widget_values = {}
    for param_name in params:
        widget = getattr(self, param_name, None)
        if widget is not None:
            if hasattr(widget, 'value'):
                widget_values[param_name] = widget.value()
            elif hasattr(widget, 'text'):
                widget_values[param_name] = widget.text()
            elif hasattr(widget, 'isChecked'):
                widget_values[param_name] = widget.isChecked()
    module.set_parameters(**widget_values)
    return module._collect_param_dict()
```

#### 1b. Replace `_run_analysis_modular()`

**Current**: Calls `module.run_batch_analysis()` and `module.save_results()` — both don't exist.

**Proposed**:
```python
def _run_analysis_modular(self, module):
    from pyAPisolation.analysis import run_batch, save_results
    
    result = run_batch(module, self.data_dir, 
                       protocol_filter=self.protocol_filter)
    
    if result.success:
        save_results(result, self.output_dir, tag=self.output_tag)
        self.display_results(result.to_dataframe())
    else:
        self.show_errors(result.errors)
```

#### 1c. Replace Individual Analysis (Preview)

**Current**: Calls `module.run_individual_analysis(abf, sweeps, params, popup)`.

**Proposed**:
```python
def _run_individual_modular(self, module, abf_data, selected_sweeps):
    result = module.run(celldata=abf_data, selected_sweeps=selected_sweeps)
    return result  # AnalysisResult with .sweep_results and .to_dataframe()
```

#### 1d. Add Progress Callback Support to Framework

Add an optional `progress_callback` parameter to `AnalysisBase.run()` and `_run_per_sweep()`:

```python
# In base.py, _run_per_sweep():
def _run_per_sweep(self, data, sweeps, result, progress_callback=None, **kwargs):
    for i, sweep_idx in enumerate(sweeps):
        if progress_callback:
            progress_callback(i, len(sweeps))
        # ... existing sweep processing ...
```

The GUI would pass:
```python
result = module.run(celldata=data, selected_sweeps=sweeps,
                    progress_callback=lambda i, n: popup.setValue(i))
```

### Priority 2: `modern_gui.py`

This file needs heavier changes because it directly instantiates `AnalysisParameters` and `AnalysisRunner`.

#### 2a. Fix Imports

```python
# OLD
from ..analysis import registry, AnalysisParameters, AnalysisRunner, AnalysisResult

# NEW
from ..analysis import registry, AnalysisBase, AnalysisResult, run_batch, save_results, get
```

#### 2b. Replace `ModernAnalysisThread`

```python
# OLD
class ModernAnalysisThread(QThread):
    def __init__(self, ..., parameters: AnalysisParameters, ...):
        self.runner = AnalysisRunner(registry)
    def run(self):
        self.results = self.runner.run_batch(
            file_pattern=..., analyzer_name=...,
            parameters=..., progress_callback=...)

# NEW
class ModernAnalysisThread(QThread):
    def __init__(self, ..., module: AnalysisBase, files_or_folder: str, ...):
        self.module = module
        self.files_or_folder = files_or_folder
    def run(self):
        self.result = run_batch(self.module, self.files_or_folder,
                                protocol_filter=self.protocol_filter)
```

#### 2c. Replace `_create_analysis_parameters()`

```python
# OLD
def _create_analysis_parameters(self) -> AnalysisParameters:
    parameters = AnalysisParameters(start_time=..., end_time=..., ...)
    analyzer = registry.get_analyzer(analysis_type)
    for key, val in analyzer.parameters.items():
        parameters.set(key, val)
    return parameters

# NEW
def _apply_gui_params_to_module(self, module: AnalysisBase):
    """Read GUI widgets and push values into the module's parameters."""
    params = {}
    params['start'] = float(self.start_time_input.text())
    params['end'] = float(self.end_time_input.text())
    # ... read other widgets ...
    module.set_parameters(**params)
```

#### 2d. Fix `_display_parameters()`

```python
# OLD — accesses Parameter objects with .value, .param_type
for name, param_info in analyzer.get_ui_elements().items():
    value = param_info.value
    param_type = param_info.param_type

# NEW — accesses plain dicts
for name, info in module.get_parameters().items():
    value = info["value"]
    param_type = info["type"]
    default = info["default"]
```

#### 2e. Fix Result Processing

```python
# OLD
result.summary_data     # DataFrame
result.detailed_data    # DataFrame  
result.analyzer_name    # str

# NEW
result.to_dataframe()   # Single combined DataFrame
result.data             # dict (file-level)
result.sweep_results    # list[dict] (sweep-level)
result.name             # str
```

### Priority 3: `example_custom_analysis.py`

Complete rewrite to serve as the new template:

```python
"""Example: how to write a custom analysis module for the GUI."""

from pyAPisolation.analysis import AnalysisBase, register
import numpy as np

class CustomStatsModule(AnalysisBase):
    """Compute basic statistics per sweep."""
    
    name = "custom_stats"
    display_name = "Custom Statistics Analysis"
    sweep_mode = "per_sweep"
    
    # Parameters — appear as GUI controls automatically
    compute_std: bool = True
    voltage_unit: str = "mV"
    
    def analyze(self, x, y, c, **kwargs):
        result = {
            "mean_voltage": float(np.mean(y)),
            "min_voltage": float(np.min(y)),
            "max_voltage": float(np.max(y)),
            "voltage_range": float(np.ptp(y)),
        }
        if self.compute_std:
            result["std_voltage"] = float(np.std(y))
        return result

# Register so GUI can discover it
register(CustomStatsModule)
```

### Priority 4: Auto-Generate GUI Controls from Parameters

Add a utility function (in `gui/` or `analysis/`) that generates PySide2 widgets from `module.get_parameters()`:

```python
def build_param_widgets(module):
    """Generate QWidget controls for each parameter."""
    from PySide2 import QtWidgets
    
    widgets = {}
    for name, info in module.get_parameters().items():
        ptype = info["type"]
        default = info["value"]
        
        if ptype is bool:
            w = QtWidgets.QCheckBox(name)
            w.setChecked(default)
        elif ptype is float:
            w = QtWidgets.QDoubleSpinBox()
            w.setValue(default)
            w.setRange(-1e9, 1e9)
        elif ptype is int:
            w = QtWidgets.QSpinBox()
            w.setValue(default)
            w.setRange(-999999, 999999)
        elif ptype is str:
            w = QtWidgets.QLineEdit(str(default))
        else:
            w = QtWidgets.QLineEdit(str(default))
        
        widgets[name] = w
    return widgets
```

This would allow the GUI to dynamically render controls for **any** registered module without hardcoded UI.

---

## Test Results

All 24 tests pass on Python 3.8.18:

```
tests/test_analysis_framework.py::TestAnalysisResult::test_basic_creation PASSED
tests/test_analysis_framework.py::TestAnalysisResult::test_add_error_marks_failure PASSED
tests/test_analysis_framework.py::TestAnalysisResult::test_to_dataframe_single PASSED
tests/test_analysis_framework.py::TestAnalysisResult::test_to_dataframe_sweep_results PASSED
tests/test_analysis_framework.py::TestAnalysisResult::test_concatenate PASSED
tests/test_analysis_framework.py::TestAnalysisResult::test_concatenate_empty PASSED
tests/test_analysis_framework.py::TestAnalysisBase::test_param_discovery PASSED
tests/test_analysis_framework.py::TestAnalysisBase::test_param_override_in_constructor PASSED
tests/test_analysis_framework.py::TestAnalysisBase::test_set_parameters PASSED
tests/test_analysis_framework.py::TestAnalysisBase::test_collect_param_dict PASSED
tests/test_analysis_framework.py::TestAnalysisBase::test_auto_name PASSED
tests/test_analysis_framework.py::TestAnalysisBase::test_run_per_sweep_with_arrays PASSED
tests/test_analysis_framework.py::TestAnalysisBase::test_run_per_file_mode PASSED
tests/test_analysis_framework.py::TestAnalysisBase::test_run_handles_exceptions PASSED
tests/test_analysis_framework.py::TestAnalysisBase::test_selected_sweeps PASSED
tests/test_analysis_framework.py::TestRegistry::test_register_and_get PASSED
tests/test_analysis_framework.py::TestRegistry::test_list_modules PASSED
tests/test_analysis_framework.py::TestRegistry::test_register_with_overrides PASSED
tests/test_analysis_framework.py::TestWithDemoData::test_spike_analysis_runs PASSED
tests/test_analysis_framework.py::TestWithDemoData::test_subthreshold_analysis_runs PASSED
tests/test_analysis_framework.py::TestWithDemoData::test_peak_detector_runs PASSED
tests/test_analysis_framework.py::TestWithDemoData::test_run_batch PASSED
tests/test_analysis_framework.py::TestLegacyImports::test_feature_extractor_still_importable PASSED
tests/test_analysis_framework.py::TestLegacyImports::test_dataset_still_importable PASSED
======================== 24 passed, 4 warnings in 4.41s ========================
```
