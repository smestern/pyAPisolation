---
name: add-analysis-module
description: "Scaffold a new electrophysiology analysis module for pyAPisolation: create an AnalysisBase subclass in pyAPisolation/analysis/builtins/, auto-register it, and add a mirroring test. Use when the user asks to add a custom analysis, new feature extractor, new metric, or any pipeline that processes ABF/NWB sweeps through the modular framework."
---

# Add Analysis Module

Scaffolds a new `AnalysisBase` subclass following the conventions in [analysis-framework.instructions.md](../../instructions/analysis-framework.instructions.md). Reference: [ANALYSIS_REFACTOR.md](../../../ANALYSIS_REFACTOR.md), example: [pyAPisolation/analysis/builtins/example.py](../../../pyAPisolation/analysis/builtins/example.py).

## When to use

- User says: "add an analysis", "new feature extractor", "custom metric", "module for X", "extract Y from sweeps".
- A repeatable single-file ABF/NWB metric is needed.
- A wrapper over an existing legacy function (`featureExtractor.*`, `patch_subthres.*`) is needed.

## When NOT to use

- Modifying [featureExtractor.py](../../../pyAPisolation/featureExtractor.py) directly — it's frozen. Wrap it instead.
- One-off scripts — put those in `pyAPisolation/dev/` or `notebooks/`.
- GUI-only changes — see [gui-migration.instructions.md](../../instructions/gui-migration.instructions.md).

## Procedure

### 1. Interview

Ask the user (use the ask-questions tool if available):

1. **Module name** (snake_case, unique). Will be the `name` class attribute and the registry key.
2. **Display name** — human label for GUIs.
3. **`sweep_mode`**: `per_sweep` (framework iterates, you get 1-D arrays) or `per_file` (you get 2-D `(sweeps × samples)` arrays and iterate yourself).
4. **Parameters** with default values and types (each becomes a typed class attribute, e.g. `threshold: float = -20.0`).
5. **Output keys** — what columns should appear in the result DataFrame.
6. **Wrapping a legacy function?** If yes, which one (e.g. `featureExtractor.analyze_sweep`, `patch_subthres.subthres_a`).

### 2. Create the module file

Path: `pyAPisolation/analysis/builtins/<module_name>.py`

Template:

```python
"""<one-line purpose>."""

import numpy as np
from ..base import AnalysisBase


class <ClassName>(AnalysisBase):
    """<docstring describing inputs, outputs, assumptions>."""

    name = "<module_name>"
    display_name = "<Display Name>"
    sweep_mode = "per_sweep"  # or "per_file"

    # Parameters — typed class attributes only
    <param>: <type> = <default>

    def analyze(self, x, y, c, **kwargs):
        # kwargs may include: sweep_number, file_path, celldata
        # Return a FLAT dict — keys become DataFrame columns
        return {"<output_key>": <value>}
```

If wrapping a legacy function, import inside `analyze()` to avoid forcing IPFX on framework import:

```python
def analyze(self, x, y, c, **kwargs):
    from ...featureExtractor import analyze_sweep
    spikes, train = analyze_sweep(x, y, c, param_dict=self._collect_param_dict())
    return {"spike_count": len(spikes), "first_isi": float(train["isi"].iloc[0]) if len(train) else np.nan}
```

### 3. Auto-register

Edit [pyAPisolation/analysis/builtins/__init__.py](../../../pyAPisolation/analysis/builtins/__init__.py) and add:

```python
from .<module_name> import <ClassName>
from ..registry import register
register(<ClassName>)
```

Verify the existing pattern in that file — match it exactly.

### 4. Add a test

Append a test class to [tests/test_analysis_framework.py](../../../tests/test_analysis_framework.py) or create a sibling `tests/test_<module_name>.py`. Minimum coverage:

```python
class Test<ClassName>:
    def test_registered(self):
        from pyAPisolation.analysis import get
        assert get("<module_name>") is not None

    def test_run_synthetic(self):
        from pyAPisolation.analysis.builtins.<module_name> import <ClassName>
        x, y, c = _make_fake_sweep(spike=True)  # helper at top of test file
        result = <ClassName>().run(x=x, y=y, c=c)
        df = result.to_dataframe()
        assert "<output_key>" in df.columns

    @pytest.mark.skipif(not HAS_DEMO_DATA, reason="demo data not present")
    def test_run_demo_file(self):
        from pyAPisolation.analysis.builtins.<module_name> import <ClassName>
        result = <ClassName>().run(file=DEMO_ABF_1)
        assert result.success
```

Use `_make_fake_sweep` / `_make_fake_data_2d` from the existing test file for IPFX-free coverage.

### 5. Validate

```powershell
pytest tests/test_analysis_framework.py -v
pytest tests/ -k "<module_name>" -v
```

If the new module shadows a legacy name, also run:

```powershell
pytest tests/test_feature_extractor.py
```

### 6. Report back

Summarize:
- File path of the new module
- Registry name and any parameters
- Test command that passed
- Any legacy function wrapped (so the user knows what to preserve)

## Checklist

- [ ] `name` is unique (`pyAPisolation.analysis.list_modules()` doesn't already include it)
- [ ] All parameters have type annotations and defaults
- [ ] `analyze()` returns a flat dict
- [ ] `sweep_mode` is `"per_sweep"` or `"per_file"` and matches input handling
- [ ] Registered in `builtins/__init__.py`
- [ ] Test passes
- [ ] No new top-level imports of IPFX or other optional deps in `analysis/`
- [ ] Legacy `featureExtractor.py` untouched
