---
description: "Use when editing gigaseal/gui/ — the canonical GUI is the dockable app.py + panels/ + controllers/ stack on the analysis.AnalysisBase framework; spikeFinder.py is frozen legacy. Covers framework usage, Qt threading, and ProcessPoolExecutor constraints."
applyTo: "gigaseal/gui/**"
---

# GUI Rules

The canonical GUI is the new dockable app built on the modular analysis framework. The previous single-window GUI is preserved but frozen. See [ANALYSIS_REFACTOR.md](../../ANALYSIS_REFACTOR.md) for the framework design — its "Proposed GUI Updates" section now describes the pattern `app.py` already follows.

## Current state

- [app.py](../../gigaseal/gui/app.py) + [panels/](../../gigaseal/gui/panels) + [controllers/analysis_controller.py](../../gigaseal/gui/controllers/analysis_controller.py) — **canonical GUI**, already on the modular framework (`AnalysisController` → `module.run` / `run_batch` → `AnalysisResult`). All new GUI feature work goes here. Launch via `python -m gigaseal.gui.app`.
- [spikeFinder.py](../../gigaseal/gui/spikeFinder.py) — **legacy, preserved.** Do not edit for feature work. Its modular code path is dead and will be removed with the file once `app.py` is verified at parity. Still reachable via the `spike_finder` console-script for regression checks. Only acceptable edits: fixes for outright breakage (import errors, crashes on launch).
- [example_custom_analysis.py](../../gigaseal/gui/example_custom_analysis.py) — uses the old API; rewrite against the modular framework if you touch it.
- `modern_gui.py` (if present) — directly instantiates removed classes `AnalysisParameters` / `AnalysisRunner`. Imports will fail at runtime; treat as dead code.

## Rules when editing the GUI

- **Do not edit `spikeFinder.py` for feature work.** Port the change into `app.py` / `panels/` / `controllers/` instead. The `spike_finder` console-script entry stays working in the meantime.
- **Read widget values through `module.get_parameters()`**, which returns `{name: {type, default, value}}` dicts — not `Parameter` objects with `.value` / `.param_type`.
- **Batch runs go through `run_batch(module, folder, ...)`**, the standalone function from `gigaseal.analysis`. Do not call `module.run_batch_analysis(...)` (removed).
- **Single-file previews use `module.run(celldata=data, selected_sweeps=...)`** and consume `AnalysisResult.to_dataframe()` / `.sweep_results`. Never `module.run_individual_analysis(...)` (removed).
- **Progress callbacks** are passed as a `progress_callback=lambda i, n: ...` kwarg into `run_batch` / `module.run`. Do not wire Qt signals directly into the framework — wrap them in a plain callable.
- **Qt threading:** any framework call inside a `QThread` must not touch GUI widgets directly. Emit signals; let the main thread update widgets. `ProcessPoolExecutor` (`n_jobs > 1` in `run_batch`) is incompatible with bound Qt methods — use module-level functions only.
- **Imports:** prefer `from gigaseal.analysis import AnalysisBase, run_batch, save_results, get, list_modules, AnalysisResult`. Do **not** import `AnalysisParameters`, `AnalysisRunner`, `AnalysisModule`, `Parameter` — these were removed (renamed to `_legacy_*`).
- **Optional deps are GUI-only.** PySide6, pyqtgraph, seaborn, prismWriter belong to the `[gui]` extra. Never import them at top level of `gigaseal/` core modules.

## Verifying a GUI change

```bash
pytest tests/test_gui_imports.py             # smoke test: imports work without crashing
python -m gigaseal.gui.app              # launch the canonical GUI and exercise the affected path
spike_finder                                  # legacy GUI — launch only for regression checks
```

The import test does not exercise behavior — interactive verification against `python -m gigaseal.gui.app` is required for any modular-path change.
