# Copilot Iteration Priorities

A working backlog for high-ROI Copilot work on gigaseal, ordered by leverage. The guiding principle: **humans write the core analysis; Copilot handles frameworking, generalization, and cleanup.** See the policy note in [README.md](README.md) and [AGENTS.md](AGENTS.md).

## Tier 1 — Finish what's half-built (biggest payoff)

1. **Finish the new dockable GUI ([gui/app.py](gigaseal/gui/app.py)).** This is now the canonical GUI and the migration target for the analysis refactor. `app.py` + [panels/](gigaseal/gui/panels) + [controllers/analysis_controller.py](gigaseal/gui/controllers/analysis_controller.py) already wire the modular path end-to-end (`AnalysisController` → `module.run` / `run_batch` → `AnalysisResult` → table/plot). Remaining work is parity polish, not plumbing: full param-form coverage for every registered module, batch progress reporting through `progress_callback`, span-selection → param round-trip edge cases, results-panel export, and an end-to-end smoke test that loads a demo ABF and runs both `SpikeAnalysis` and `SubthresholdAnalysis` through `AnalysisController`. The [gui-migration instruction](.github/instructions/gui-migration.instructions.md) is the guardrail.

    *Legacy note:* [spikeFinder.py](gigaseal/gui/spikeFinder.py) is the previous single-window GUI. It is **preserved as-is** — no new feature work, no further modular-path retrofitting. It stays reachable via the `spike_finder` console-script entry point for regression checks until `app.py` is verified at parity, then it will be deprecated and removed. Any feature request that would have landed in `spikeFinder.py` should land in `app.py`/`panels/`/`controllers/` instead.

## Tier 2 — Consolidation that pays back forever

3. **Unify `bin/run_*.py` scripts behind a single argparse/click CLI.** Eight scripts with copy-pasted tkinter prompt blocks; collapse to `gigaseal <subcommand>` over the new `run_batch` API. Mechanical, repetitive, perfect Copilot work. Also makes future analyses one-line CLI-available.

4. **Test scaffolding for every builtin.** Each module in `gigaseal/analysis/builtins/` should have a `Test<Class>` block in [tests/test_analysis_framework.py](tests/test_analysis_framework.py) covering: registered, synthetic sweep, demo ABF (skipif). The [`/add-analysis-module` skill](.github/skills/add-analysis-module/SKILL.md) sets the pattern — point Copilot at it and let it fill gaps for `SpikeAnalysis` and `SubthresholdAnalysis`.

## Tier 3 — Hygiene

5. **Delete stale `build/` Dash code and `_legacy_*` files** once you're sure the new framework is wired in. One PR, big footprint reduction. **Not in this set:** [spikeFinder.py](gigaseal/gui/spikeFinder.py) — it stays until `app.py` reaches parity and is signed off (see Tier 1).
6. **CI matrix** — the only workflow is `.github/workflows/python-package-conda.yml`. Add a fast pytest job (matrix over 3.11/3.12) and a ruff/lint job. Copilot generates these in seconds.

## Skip for now (until budget recovers)

- **Web viz refactor** — Flask works; the Dash fork in `build/` is noted as abandoned.
- **`dev/` experiments** — by definition not load-bearing.
- **Algorithm / IPFX work** — your lane, keep it there.

## Concrete next session

Single focused prompt:

> Close parity gaps in `gigaseal/gui/app.py` against the modular analysis framework — full param-form coverage for every registered module, batch `progress_callback` wired into the status bar, results export round-trip — and extend `tests/test_gui_imports.py` with an end-to-end smoke test that drives `AnalysisController` against a demo ABF for both `SpikeAnalysis` and `SubthresholdAnalysis`. Do not touch `spikeFinder.py`.

The gui-migration instruction will auto-attach. That PR closes the loop on the refactor and unblocks deletion of the `_legacy_*` files (and eventually `spikeFinder.py` itself).

## Division-of-labor reminder

| Layer | Author |
|---|---|
| Core spike / subthreshold algorithms ([featureExtractor.py](gigaseal/featureExtractor.py), [patch_subthres.py](gigaseal/patch_subthres.py)) | **Human** |
| IPFX integration, Inoue-lab IC1 logic | **Human** |
| New `AnalysisBase` subclasses (the `analyze()` body) | **Human** — Copilot may scaffold |
| Framework plumbing, GUI glue, CLI, registry wiring, batch loops | **Copilot** |
| Tests, type hints, docstrings, refactors, cleanup of `_legacy_*` and `build/` | **Copilot** |
| CI, packaging, project structure | **Copilot** |
