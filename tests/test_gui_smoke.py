"""
End-to-end smoke test for the new dockable GUI (app.py).

Exercises the modular path through AnalysisController against a demo
ABF for both SpikeAnalysis and SubthresholdAnalysis, plus a
ResultsPanel CSV export round-trip.
"""

import os
import sys

import pandas as pd
import pytest

# Headless Qt — must be set before any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PySide6 = pytest.importorskip("PySide6")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DEMO_ABF = os.path.abspath(os.path.join(DATA_DIR, "demo_data_1.abf"))
HAS_DEMO = os.path.exists(DEMO_ABF)


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture(scope="module")
def main_window(qapp):
    from gigaseal.gui.app import MainWindow
    w = MainWindow()
    yield w
    w.close()


# ---------------------------------------------------------------------------
# Phase D — param-form coverage audit
# ---------------------------------------------------------------------------

def test_param_form_covers_all_visible_module_types():
    """Every visible module's parameter types must be handled by ParamFormWidget."""
    from gigaseal.analysis import get_all

    supported = {float, int, bool, str}
    registry = get_all()
    offenders = []
    for name, cls in registry.items():
        inst = cls() if isinstance(cls, type) else cls
        if getattr(inst, "hidden", False):
            continue
        for pname, info in inst.get_parameters().items():
            ptype = info.get("type")
            if ptype not in supported:
                offenders.append(f"{name}.{pname}: {ptype}")
    assert not offenders, (
        f"ParamFormWidget does not handle these parameter types: {offenders}"
    )


def test_legacy_spike_is_hidden_from_panel(qapp):
    """LegacySpikeAnalysis must not appear in the AnalysisPanel combo."""
    from gigaseal.gui.panels.analysis_panel import AnalysisPanel
    panel = AnalysisPanel()
    try:
        items = [panel._combo_module.itemText(i)
                 for i in range(panel._combo_module.count())]
        assert "Legacy Spike Analysis" not in items
        # at least one visible module should be present
        assert items, "AnalysisPanel combo is empty"
    finally:
        panel.deleteLater()


# ---------------------------------------------------------------------------
# Phase E — end-to-end against a demo ABF
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not HAS_DEMO, reason="demo_data_1.abf not found")
def test_controller_load_file(main_window):
    cd = main_window._controller.load_file(DEMO_ABF)
    assert cd is not None
    assert cd.sweepCount > 0


@pytest.mark.skipif(not HAS_DEMO, reason="demo_data_1.abf not found")
@pytest.mark.parametrize("module_name", ["spike", "subthreshold"])
def test_individual_analysis_modular(main_window, module_name):
    """Run module.run() directly on the loaded cellData — modular path only."""
    from gigaseal.analysis import get

    main_window._controller.load_file(DEMO_ABF)
    cd = main_window._controller.celldata
    module = get(module_name)
    assert module is not None, f"{module_name} not registered"

    result = module.run(celldata=cd, selected_sweeps=list(range(cd.sweepCount)))
    assert result.success, f"{module_name} errors: {result.errors}"
    df = result.to_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty, f"{module_name} produced empty DataFrame"


@pytest.mark.skipif(not HAS_DEMO, reason="demo_data_1.abf not found")
@pytest.mark.parametrize("module_name", ["spike", "subthreshold"])
def test_batch_analysis_modular(tmp_path, module_name):
    """Run runner.run_batch directly against data/ folder for determinism."""
    from gigaseal.analysis import get, run_batch, save_results

    module = get(module_name)
    folder = os.path.abspath(DATA_DIR)

    progress_calls = []

    def cb(done, total):
        progress_calls.append((done, total))

    result = run_batch(module, folder, n_jobs=1, progress_callback=cb)
    assert result is not None
    # progress callback should fire at least once per processed file
    assert progress_calls, "progress_callback was never invoked"
    assert progress_calls[-1][0] == progress_calls[-1][1]

    # save_results should write a CSV
    out_path = save_results(result, str(tmp_path), tag="smoke", fmt="csv")
    assert os.path.exists(out_path)


# ---------------------------------------------------------------------------
# Phase C — ResultsPanel export / import round-trip
# ---------------------------------------------------------------------------

def test_results_panel_csv_roundtrip(qapp, tmp_path):
    from gigaseal.gui.panels.results_panel import ResultsPanel

    panel = ResultsPanel()
    try:
        df = pd.DataFrame({
            "filename": ["a.abf", "b.abf"],
            "spike_count": [3, 7],
            "rheobase": [0.05, 0.10],
        })
        panel.set_dataframe(df, index_col=None)

        out = tmp_path / "results.csv"
        panel.get_dataframe().to_csv(out, index=False)
        assert out.exists()

        reloaded = pd.read_csv(out)
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True),
            reloaded.reset_index(drop=True),
        )
    finally:
        panel.deleteLater()


def test_results_panel_set_sheets_tabs(qapp):
    """set_sheets builds one tab per non-empty sheet and skips empties."""
    from collections import OrderedDict
    from gigaseal.gui.panels.results_panel import ResultsPanel

    panel = ResultsPanel()
    try:
        sheets = OrderedDict([
            ("Summary", pd.DataFrame({"filename": ["a.abf"], "n_sweeps": [3]})),
            ("Raw", pd.DataFrame({"filename": ["a.abf", "a.abf"], "spike_count": [1, 2]})),
            ("Empty", pd.DataFrame()),
        ])
        panel.set_sheets(sheets, index_col=None)

        assert panel._tabs.count() == 2
        labels = [panel._tabs.tabText(i) for i in range(panel._tabs.count())]
        assert labels == ["Summary", "Raw"]

        # Active-tab DataFrame and full sheet round-trip
        panel._tabs.setCurrentIndex(0)
        assert "n_sweeps" in panel.get_dataframe().columns
        assert set(panel.get_sheets().keys()) == {"Summary", "Raw"}
    finally:
        panel.deleteLater()

