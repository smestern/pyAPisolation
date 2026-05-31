---
description: "Use when authoring or modifying analysis modules in pyAPisolation/analysis/ — covers AnalysisBase contract, parameter discovery, sweep_mode semantics, and registry patterns. Apply when subclassing AnalysisBase, registering modules, or touching builtins."
applyTo: "pyAPisolation/analysis/**"
---

# Analysis Framework Rules

The modular analysis framework lives in [pyAPisolation/analysis/](../../pyAPisolation/analysis/). See [ANALYSIS_REFACTOR.md](../../ANALYSIS_REFACTOR.md) for the full migration guide; the canonical test reference is [tests/test_analysis_framework.py](../../tests/test_analysis_framework.py).

## Hard rules

- **Parameters are typed class attributes**, never `Parameter` objects:
  ```python
  class MyAnalysis(AnalysisBase):
      name = "my_analysis"
      sweep_mode = "per_sweep"
      dv_cutoff: float = 7.0          # ✅ discovered via __annotations__
      # dv_cutoff = Parameter(...)    # ❌ legacy API — do not reintroduce
  ```
  The framework discovers them via `__annotations__`. A parameter without a type annotation is invisible.

- **`name` must be unique** in the global registry. Lower-case, snake_case. If omitted, defaults to `cls.__name__.lower()`.

- **`sweep_mode` determines `analyze()` input shape:**
  | Mode | `x`, `y`, `c` shape | Framework iterates? |
  |---|---|---|
  | `"per_sweep"` | 1-D (single sweep) | Yes — once per sweep |
  | `"per_file"` | 2-D `(sweeps × samples)` | No — your code iterates |

- **`analyze()` must return a flat `dict`.** Nested dicts/lists become object columns and break `to_dataframe()`. Keys become column names.

- **Framework-injected kwargs** in `per_sweep` mode: `sweep_number`, `file_path`, `celldata`. Accept via `**kwargs` and never make them positional.

- **Register at module level**, not inside functions or `if __name__ == ...`. `run_batch(n_jobs>1)` uses `ProcessPoolExecutor` on Windows — modules must be importable in worker processes.

- **Built-ins delegate to legacy code.** Wrappers in [pyAPisolation/analysis/builtins/](../../pyAPisolation/analysis/builtins/) call into `featureExtractor.py` / `patch_subthres.py`. Do not duplicate logic — wrap it.

- **Legacy `featureExtractor.py` is frozen.** Never change its public function signatures (`analyze`, `analyze_sweep`, `analyze_sweepset`, `batch_feature_extract`) — the GUI and bin scripts still depend on them.

## Patterns

**Reference implementation** (10 lines of user code): [pyAPisolation/analysis/builtins/example.py](../../pyAPisolation/analysis/builtins/example.py).

**Runtime parameter override:**
```python
module = MyAnalysis(dv_cutoff=10.0)             # constructor
module.set_parameters(dv_cutoff=12.0)           # post-hoc
```

**Batch entry point** — always use the standalone function, never a method:
```python
from pyAPisolation.analysis import run_batch, save_results
result = run_batch(module, "folder/", protocol_filter="IC1", n_jobs=4)
save_results(result, "out/", tag="exp1")
```

## Tests

Every new module needs a test in [tests/test_analysis_framework.py](../../tests/test_analysis_framework.py) (or a sibling file) that exercises both `module.run(file=...)` and `run_batch(module, folder)` using fixtures from `data/`. The `_make_fake_sweep` / `_make_fake_data_2d` helpers cover the no-IPFX case.

## Anti-patterns

- Re-creating `AnalysisModule`, `AnalysisParameters`, `Parameter`, or `AnalysisRunner` (all removed; renamed to `_legacy_*`).
- Calling `module.run_batch_analysis(...)` / `module.save_results(...)` / `module.get_ui_elements()` / `module.parse_ui_params(...)` — these are pre-refactor methods. Use the new equivalents listed in the migration table in [ANALYSIS_REFACTOR.md](../../ANALYSIS_REFACTOR.md).
- Importing IPFX at module top level in `analysis/builtins/*` — keep imports inside `analyze()` or guard them, so the framework remains importable without optional deps.
