# Copilot Iteration Priorities

A working backlog for high-ROI Copilot work on pyAPisolation, ordered by leverage. The guiding principle: **humans write the core analysis; Copilot handles frameworking, generalization, and cleanup.** See the policy note in [README.md](README.md) and [AGENTS.md](AGENTS.md).

## Tier 1 — Finish what's half-built (biggest payoff)

1. **GUI migration in [spikeFinder.py](pyAPisolation/gui/spikeFinder.py)** — [ANALYSIS_REFACTOR.md](ANALYSIS_REFACTOR.md) already specifies every replacement (sections 1a–1d, 2a–2e). It's pure plumbing: widget → `module.set_parameters` → `run_batch` → DataFrame display. Currently the modular path is dead code that silently falls back to legacy. Copilot can chew through this with the new [gui-migration instruction](.github/instructions/gui-migration.instructions.md) as a guardrail, and you get a real end-to-end test of the framework.

2. **Triage `modern_gui.py`** — if it still imports removed `AnalysisParameters` / `AnalysisRunner`, it's broken on import. Decide: delete it, or migrate it. One short Copilot session.

## Tier 2 — Consolidation that pays back forever

3. **Unify `bin/run_*.py` scripts behind a single argparse/click CLI.** Eight scripts with copy-pasted tkinter prompt blocks; collapse to `pyAPisolation <subcommand>` over the new `run_batch` API. Mechanical, repetitive, perfect Copilot work. Also makes future analyses one-line CLI-available.

4. **Test scaffolding for every builtin.** Each module in `pyAPisolation/analysis/builtins/` should have a `Test<Class>` block in [tests/test_analysis_framework.py](tests/test_analysis_framework.py) covering: registered, synthetic sweep, demo ABF (skipif). The [`/add-analysis-module` skill](.github/skills/add-analysis-module/SKILL.md) sets the pattern — point Copilot at it and let it fill gaps for `SpikeAnalysis` and `SubthresholdAnalysis`.

## Tier 3 — Hygiene

5. **Delete stale `build/` Dash code and `_legacy_*` files** once you're sure the new framework is wired in. One PR, big footprint reduction.
6. **CI matrix** — the only workflow is `.github/workflows/python-package-conda.yml`. Add a fast pytest job (matrix over 3.11/3.12) and a ruff/lint job. Copilot generates these in seconds.

## Skip for now (until budget recovers)

- **Web viz refactor** — Flask works; the Dash fork in `build/` is noted as abandoned.
- **`dev/` experiments** — by definition not load-bearing.
- **Algorithm / IPFX work** — your lane, keep it there.

## Concrete next session

Single focused prompt:

> Migrate `spikeFinder.py` to the new analysis framework per `ANALYSIS_REFACTOR.md` sections 1a–1d, keeping the legacy fallback intact, and add a smoke test.

The gui-migration instruction will auto-attach. That single PR closes the loop on the refactor and unblocks deletion of the `_legacy_*` files later.

## Division-of-labor reminder

| Layer | Author |
|---|---|
| Core spike / subthreshold algorithms ([featureExtractor.py](pyAPisolation/featureExtractor.py), [patch_subthres.py](pyAPisolation/patch_subthres.py)) | **Human** |
| IPFX integration, Inoue-lab IC1 logic | **Human** |
| New `AnalysisBase` subclasses (the `analyze()` body) | **Human** — Copilot may scaffold |
| Framework plumbing, GUI glue, CLI, registry wiring, batch loops | **Copilot** |
| Tests, type hints, docstrings, refactors, cleanup of `_legacy_*` and `build/` | **Copilot** |
| CI, packaging, project structure | **Copilot** |
