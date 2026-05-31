# gigaseal Analysis Framework Refactor — Changelog & Migration Guide

> **Status (2026-05-31):** Framework core is shipped and stable. GUI migration (Priorities 1–4 below) is **not started** — the GUI still relies on legacy `featureExtractor` shims and silently falls back to legacy code paths.

## Overview

The `gigaseal.analysis` submodule was rebuilt from scratch to be **simple, modular, and user-friendly**. The old framework (`AnalysisModule`, `AnalysisParameters`, `Parameter`, `AnalysisRegistry`, `AnalysisRunner`) has been replaced with a single base class, a flat function registry, and a standalone batch runner.

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
| `gigaseal/analysis/result.py` | `AnalysisResult` dataclass — lightweight result container with `to_dataframe()` and `concatenate()` |
| `gigaseal/analysis/base.py` | `AnalysisBase` — the single base class; handles input normalization, sweep iteration, parameter introspection |
| `gigaseal/analysis/registry.py` | `register()`, `get()`, `list_modules()`, `get_all()`, `clear()` — flat module registry |
| `gigaseal/analysis/runner.py` | `run_batch()`, `save_results()` — batch processing with optional `ProcessPoolExecutor` and `progress_callback` |
| `gigaseal/analysis/analysis_configs.json` | Parameter presets (`spike_analysis_standard`, `spike_analysis_sensitive`, `subthreshold_standard`, `subthreshold_detailed`) |
| `gigaseal/analysis/builtins/__init__.py` | Auto-registers built-in modules on import |
| `gigaseal/analysis/builtins/spike.py` | `SpikeAnalysis` (per-sweep, wraps `featureExtractor.analyze_sweep()`) and `LegacySpikeAnalysis` (per-file, wraps legacy `featureExtractor.analyze()`) |
| `gigaseal/analysis/builtins/subthreshold.py` | `SubthresholdAnalysis` — wraps `patch_subthres.subthres_a()` |
| `gigaseal/analysis/builtins/example.py` | `PeakDetector` — minimal example for new users (~10 lines of user code); demo only, not auto-registered |
| `gigaseal/analysis/__init__.py` | Public API exports + backward-compatible shims for GUI (see below) |
| `tests/test_analysis_framework.py` | 26 tests covering results, base class, registry, demo data integration (incl. `LegacySpikeAnalysis`), and legacy imports |

### Old Files Removed

The pre-refactor implementations (`base.py`, `builtin_modules.py`, `registry.py`, `runner.py`, `utilities.py`, and the old `__init__.py`) were **deleted outright** rather than renamed to `_legacy_*`. There is no in-tree fallback copy of the old framework. Earlier drafts of this document described a `_legacy_*.py` preservation scheme — that scheme was abandoned in favour of a clean break, since the legacy functional API in `gigaseal/featureExtractor.py` already serves as the compatibility layer.

### Backward-Compatibility Shims (`analysis/__init__.py`)

To keep older callers from crashing on import, `analysis/__init__.py` exposes:

- Placeholder aliases: `AnalysisModule = AnalysisBase`, `AnalysisParameters = None`, `AnalysisRunner = None`.
- A `_RegistryCompat` wrapper bound as `registry` with forwarding methods: `get_module(name)` → `get(name)`, `get_analyzer(name)` → `get(name)`, `register_module(m)` → `register(m)`, `list_modules_detailed()` → `{name: display_name}`, and `get_module_by_tab(idx)` → `None` (intentional, forces legacy fallback in the GUI).

These shims exist solely to keep import sites alive during the GUI migration. New code should import from the public `__all__` (see below) and ignore the shim layer.

---

## How to Write a Custom Analysis Module

```python
from gigaseal.analysis import AnalysisBase, register
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
from gigaseal.analysis import get, run_batch, save_results

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

> **Migration status (2026-05-31):** All four priorities below are **PENDING**. The GUI files currently use backward-compatibility shims that prevent crashes but always fall back to legacy code paths.

The GUI files (`spikeFinder.py`, `example_custom_analysis.py`) currently use backward-compatibility shims that prevent crashes but effectively always fall back to legacy code paths. Below are the proposed changes to fully integrate the new framework.

### Priority 1 — PENDING: `spikeFinder.py` (Main GUI)

The spikeFinder currently "works" because every call to the modular API hits a method that does not exist on `AnalysisBase` (`get_ui_elements`, `parse_ui_params`, `run_batch_analysis`, `module.save_results`), raises `AttributeError`, and is caught by a try/except that routes to `_run_analysis_legacy()`. To actually use the new framework:

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
    from gigaseal.analysis import run_batch, save_results
    
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

`run_batch()` in `gigaseal/analysis/runner.py` already accepts a `progress_callback` argument. `AnalysisBase.run()` and `_run_per_sweep()` do **not** — single-file per-sweep progress reporting from the GUI is still blocked. Add an optional `progress_callback` parameter to `AnalysisBase.run()` and `_run_per_sweep()`:

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

### Priority 2 — OBSOLETE: `modern_gui.py`

`gigaseal/gui/modern_gui.py` was removed from the codebase before migration started. Earlier drafts of this document contained a multi-step Priority 2 plan (fix imports, replace `ModernAnalysisThread`, `_create_analysis_parameters`, `_display_parameters`, result processing) — that plan is **void**. The migration target is `spikeFinder.py` only. If a modern Qt frontend is reintroduced later, reapply the same patterns shown in Priority 1.

### Priority 3 — PENDING: `example_custom_analysis.py`

The current `gigaseal/gui/example_custom_analysis.py` still uses the pre-refactor API: it imports a nonexistent `analysis_registry`, subclasses `AnalysisModule` with positional `__init__` args, and defines `get_ui_elements`, `parse_ui_params`, `run_individual_analysis`, `run_batch_analysis`. It will fail at import time once the `analysis_registry` alias is removed. Complete rewrite to serve as the new template:

```python
"""Example: how to write a custom analysis module for the GUI."""

from gigaseal.analysis import AnalysisBase, register
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

### Priority 4 — PENDING: Auto-Generate GUI Controls from Parameters

No `build_param_widgets` utility exists yet in either `gigaseal/gui/` or `gigaseal/analysis/`. Add a utility function (in `gui/` or `analysis/`) that generates PySide2 widgets from `module.get_parameters()`:

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

All 26 tests in `tests/test_analysis_framework.py` pass on Python 3.11+ (the project's minimum supported version per `pyproject.toml`).

Coverage by test class:

| Test class | Count | Notes |
|---|---|---|
| `TestAnalysisResult` | 6 | creation, error marking, `to_dataframe()` (single + sweep), `concatenate()` (incl. empty) |
| `TestAnalysisBase` | 10 | parameter discovery / override / type coercion, `_collect_param_dict`, auto-naming, per-sweep + per-file dispatch, exception handling, sweep selection |
| `TestRegistry` | 3 | register/get, list, overrides |
| `TestWithDemoData` | 5 | `SpikeAnalysis`, `LegacySpikeAnalysis`, `SubthresholdAnalysis`, `PeakDetector`, `run_batch` end-to-end on `data/demo_data_*.abf` |
| `TestLegacyImports` | 2 | `featureExtractor` and `dataset` still importable |

Run with `pytest tests/test_analysis_framework.py -v`.

---

## Related CLI Changes

Two `bin/` entry points are now deprecated thin wrappers that emit a `DeprecationWarning` and redirect to the unified `gigaseal` CLI:

- `gigaseal/bin/run_analysis_wizard.py` → `gigaseal gui analysis-wizard`
- `gigaseal/bin/convert_json_to_yaml.py` → `gigaseal convert-config`

Both touch analysis-config flow but are otherwise independent of the framework refactor.
