"""Import verification for the new GUI modules.

Skipped cleanly when PySide6 is unavailable (e.g. headless minimal installs)
so it never aborts collection for the rest of the suite.
"""
import os

import pytest

# GUI stack requires PySide6; skip the whole module if it is not installed.
pytest.importorskip("PySide6")

# Run Qt without a display so imports that touch QApplication work in CI.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_widget_imports():
    from gigaseal.gui.widgets.pandas_model import PandasModel  # noqa: F401
    from gigaseal.gui.widgets.param_form import ParamFormWidget  # noqa: F401
    from gigaseal.gui.widgets.sweep_selector import SweepSelector  # noqa: F401


def test_panel_imports():
    from gigaseal.gui.panels.file_panel import FilePanel  # noqa: F401
    from gigaseal.gui.panels.analysis_panel import AnalysisPanel  # noqa: F401
    from gigaseal.gui.panels.results_panel import ResultsPanel  # noqa: F401
    from gigaseal.gui.panels.plot_panel import PlotPanel  # noqa: F401


def test_controller_import():
    from gigaseal.gui.controllers.analysis_controller import AnalysisController  # noqa: F401


def test_app_import():
    from gigaseal.gui.app import MainWindow, main  # noqa: F401
