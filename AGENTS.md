# AGENTS.md — gigaseal

Batch electrophysiology feature extraction for ABF/NWB files (Inoue Lab @ Western). See [README.md](README.md) for user-facing docs and [ANALYSIS_REFACTOR.md](ANALYSIS_REFACTOR.md) for the new analysis framework migration guide.

## Authoring policy (soft target)

**Core analysis code is human-authored.** That means the bodies of spike-detection routines, subthreshold fits, IPFX integration, and any Inoue-lab-specific protocol logic (e.g. IC1) should be written by a person — not generated wholesale by an AI agent. Agents may:

- Scaffold new `AnalysisBase` subclasses (registration, parameters, tests) — leave `analyze()` body for the human.
- Refactor, generalize, or migrate framework/plumbing code (GUI glue, CLI, batch loops, registry wiring).
- Write tests, type hints, docstrings, and CI config.
- Clean up `_legacy_*` files, stale `build/` artifacts, and dead code paths.

See [COPILOT_PRIORITIES.md](COPILOT_PRIORITIES.md) for the current backlog and a full division-of-labor table.

## Architecture at a glance

Data flows through four layers. Always respect these boundaries:

1. **Loaders** ([gigaseal/loadFile/__init__.py](gigaseal/loadFile/__init__.py)) — `loadFile(path)` dispatches `.abf` → `loadABF`, `.nwb` → `loadNWB`. Returns `(dataX, dataY, dataC)` arrays following pyABF conventions (time, response, command).
2. **Data container** ([gigaseal/dataset.py](gigaseal/dataset.py)) — `cellData` wraps loader output and is the canonical object passed between layers. Built from a file path *or* raw arrays.
3. **Analysis** — two parallel pipelines coexist:
   - **Legacy functional API** in [gigaseal/featureExtractor.py](gigaseal/featureExtractor.py) (`analyze()`, `analyze_sweep()`, `batch_feature_extract()`) and [gigaseal/patch_subthres.py](gigaseal/patch_subthres.py). Still actively used by GUI and bin scripts; **do not modify the public signatures**.
   - **New modular framework** in [gigaseal/analysis/](gigaseal/analysis/) — `AnalysisBase` + `register()` + `run_batch()`. Built-in wrappers in [gigaseal/analysis/builtins/](gigaseal/analysis/builtins/) delegate to the legacy functions. New analyses go here.
4. **Consumers** — GUI ([gigaseal/gui/](gigaseal/gui/)), web viz ([gigaseal/webViz/](gigaseal/webViz/)), database builder ([gigaseal/database/build_database.py](gigaseal/database/build_database.py)), and CLI entry points in [gigaseal/bin/](gigaseal/bin/).

```
ABF/NWB file → loadFile → cellData → {legacy featureExtractor | analysis.AnalysisBase}
                                   → AnalysisResult / DataFrame
                                   → save_data_frames / save_results
                                   → CSV / Excel / Prism / web viz
```

## Critical project-specific conventions

- **Two analysis APIs coexist.** The legacy `featureExtractor.py` is frozen (untouched by the refactor). New work belongs in `gigaseal/analysis/`. See the API migration table in [ANALYSIS_REFACTOR.md](ANALYSIS_REFACTOR.md).
- **`AnalysisBase` parameters are typed class attributes**, not `Parameter` objects. Example: `dv_cutoff: float = 7.0`. The framework discovers them via `__annotations__`. Never reintroduce the old `Parameter` / `AnalysisParameters` wrappers.
- **`sweep_mode` matters.** `"per_sweep"` → framework iterates and passes 1-D arrays; `"per_file"` → analyze receives 2-D `(sweeps × samples)` arrays and handles its own iteration.
- **`IC1_SPECIFIC_FUNCTIONS = True`** in [featureExtractor.py](gigaseal/featureExtractor.py) swaps in Inoue-lab-specific routines for the IC1 protocol. Touch with care.
- **IPFX import is soft.** `gigaseal/__init__.py` warns rather than fails if IPFX is missing. NWB-loader imports in `dataset.py` are wrapped in `try/except`. Preserve this pattern — users with partial installs must still import the package.
- **Optional-dependency groups** in [pyproject.toml](pyproject.toml): `gui`, `web`, `server`, `ml`, `dev`, `full`. Never add imports from these groups at top level of core modules; gate them inside the relevant submodule.
- **Web viz is Flask-only.** Any Dash/Plotly code under `build/` is stale — see [repo memory](/memories/repo/web_viz_module_architecture.md). Do not resurrect it without updating `pyproject.toml`.

## Build / test / run

- **Install (dev):** `pip install -e ".[full]"` from repo root. Requires Python 3.11+.
- **Run tests:** `pytest tests/` — key suites: `test_analysis_framework.py` (24 tests, the canonical reference for the new framework), `test_feature_extractor.py`, `test_datasets.py`, `test_webviz.py`.
- **GUI entry point:** `spike_finder` (declared in `[project.scripts]`). Source: [gigaseal/bin/run_spike_finder.py](gigaseal/bin/run_spike_finder.py).
- **CLI entry points** live in [gigaseal/bin/](gigaseal/bin/) as `run_*.py` scripts (database builder, web viz, RMP, CM_CALC, prism writer, etc.). Most are tkinter-prompt-based for interactive use.
- **PyInstaller specs** (`*.spec`) sit next to their scripts in `bin/` for frozen distribution.
- **Demo data** lives in `data/` (`demo_data_*.abf`, `2021_09_23_0037.npz`) — use these in tests rather than committing new fixtures.

## Integration pipelines (where layers meet)

| Pipeline | Entry | Touches |
|---|---|---|
| GUI spike analysis | [gui/spikeFinder.py](gigaseal/gui/spikeFinder.py) | `cellData` → `featureExtractor.analyze_sweepset` → DataFrame → Qt widgets. Currently uses backward-compat shims; full migration to `analysis.run_batch` is pending (see [ANALYSIS_REFACTOR.md](ANALYSIS_REFACTOR.md) "Proposed GUI Updates"). |
| Batch CLI | [bin/run_spike_finder_cli.py](gigaseal/bin/run_spike_finder_cli.py) | tkinter prompts → `featureExtractor.batch_feature_extract` → `save_data_frames`. |
| Database build | [database/build_database.py](gigaseal/database/build_database.py) | Folder of ABF/NWB → IPFX feature collection → CSV/JSON for web viz. |
| Web visualization | [webViz/run_web_viz.py](gigaseal/webViz/run_web_viz.py) | Auto-calls `build_database` if no DB given → `ephysDatabaseViewer.main` (static or dynamic Flask). |
| New custom analysis | subclass `AnalysisBase` → `register()` → `run_batch(module, folder)` → `save_results`. Reference: [analysis/builtins/example.py](gigaseal/analysis/builtins/example.py). |

## Gotchas

- **Path conventions:** `cellData.filePath` uses `os.path.abspath`; `fileName` uses `'/'.split()` (Unix-style). On Windows, prefer `os.path.basename` when adding new code.
- **Sweep numbering:** real sweep numbers vs. 0-indexed differ — use `patch_utils.sweepNumber_to_real_sweep_number`.
- **Stale build artifacts:** `build/` and `gigaseal.egg-info/` may contain outdated copies of modules (notably an experimental Dash web_viz). Source of truth is always `gigaseal/`.
- **Python version annotations:** project targets 3.11+ but some legacy modules avoided `tuple[...]` annotations — already fixed in `patch_subthres.py`. Use PEP 604 freely in new code.
- **Multiprocessing on Windows:** `run_batch(n_jobs>1)` uses `ProcessPoolExecutor`; analysis modules must be importable at module level (no closures).
