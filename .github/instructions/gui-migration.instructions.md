---
description: "Use when editing pyAPisolation/gui/ — covers the pending migration from legacy featureExtractor shims to the new analysis.AnalysisBase framework, backward-compat fallback patterns, and Qt threading rules."
applyTo: "pyAPisolation/gui/**"
---

# GUI Migration Rules

The GUI is mid-migration from the legacy functional API to the new analysis framework. See the "Proposed GUI Updates" section of [ANALYSIS_REFACTOR.md](../../ANALYSIS_REFACTOR.md) for the full plan.

## Current state

- [spikeFinder.py](../../pyAPisolation/gui/spikeFinder.py) — main app. Calls modular helpers but every modular call currently throws `AttributeError` and falls back to legacy `featureExtractor.analyze_sweepset`. **Functional today, but the modular path is dead code.**
- [example_custom_analysis.py](../../pyAPisolation/gui/example_custom_analysis.py) — uses the old API; must be rewritten when fully migrating.
- `modern_gui.py` (if present) — directly instantiates removed classes `AnalysisParameters` / `AnalysisRunner`. Imports will fail at runtime.

## Rules when editing the GUI

- **Do not remove the legacy fallbacks** in `spikeFinder.py` until the modular path is verified end-to-end (read widget → `module.set_parameters` → `run_batch` → display DataFrame). Removing them now bricks the GUI.
- **Read widget values through `module.get_parameters()`**, which returns `{name: {type, default, value}}` dicts — not `Parameter` objects with `.value` / `.param_type`. See the proposed `_apply_gui_params_to_module()` and `_display_parameters()` rewrites in [ANALYSIS_REFACTOR.md](../../ANALYSIS_REFACTOR.md).
- **Batch runs go through `run_batch(module, folder, ...)`**, the standalone function from `pyAPisolation.analysis`. Do not call `module.run_batch_analysis(...)` (removed).
- **Single-file previews use `module.run(celldata=data, selected_sweeps=...)`** and consume `AnalysisResult.to_dataframe()` / `.sweep_results`. Never `module.run_individual_analysis(...)` (removed).
- **Progress callbacks** are passed as a `progress_callback=lambda i, n: ...` kwarg into `run_batch` / `module.run`. Do not wire Qt signals directly into the framework — wrap them in a plain callable.
- **Qt threading:** any framework call inside a `QThread` must not touch GUI widgets directly. Emit signals; let the main thread update widgets. `ProcessPoolExecutor` (`n_jobs > 1` in `run_batch`) is incompatible with bound Qt methods — use module-level functions only.
- **Imports:** prefer `from pyAPisolation.analysis import AnalysisBase, run_batch, save_results, get, list_modules, AnalysisResult`. Do **not** import `AnalysisParameters`, `AnalysisRunner`, `AnalysisModule`, `Parameter` — these were removed (renamed to `_legacy_*`).
- **Optional deps are GUI-only.** PySide6, pyqtgraph, seaborn, prismWriter belong to the `[gui]` extra. Never import them at top level of `pyAPisolation/` core modules.

## Verifying a GUI change

```bash
pytest tests/test_gui_imports.py     # smoke test: imports work without crashing
spike_finder                          # launch and exercise the affected path manually
```

The import test does not exercise behavior — interactive verification is required for any modular-path change.
