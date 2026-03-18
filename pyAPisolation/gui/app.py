"""
pyAPisolation — New Dockable GUI
=================================

A PySide6 QMainWindow with dockable panels for patch-clamp
electrophysiology analysis.  Supports both matplotlib and pyqtgraph
plot backends, auto-generated parameter forms from the analysis
registry, and legacy fallback for built-in spike / subthreshold
modules.

Launch
------
::

    python -m pyAPisolation.gui.app
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
    QMessageBox,
    QStatusBar,
)

from .panels.file_panel import FilePanel
from .panels.analysis_panel import AnalysisPanel
from .panels.results_panel import ResultsPanel
from .panels.plot_panel import PlotPanel
from .controllers.analysis_controller import AnalysisController


class MainWindow(QMainWindow):
    """
    Top-level window.

    Layout
    ------
    - **Central**: PlotPanel (trace viewer)
    - **Left dock**: FilePanel (folder browser + sweep selector)
    - **Right dock**: AnalysisPanel (module selector + params + run buttons)
    - **Bottom dock**: ResultsPanel (sortable table + export)
    """

    APP_TITLE = "pyAPisolation — Patch-Clamp Analysis"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.APP_TITLE)
        self.resize(1400, 900)

        # ---- Panels ----
        self._plot_panel = PlotPanel()
        self._file_panel = FilePanel()
        self._analysis_panel = AnalysisPanel()
        self._results_panel = ResultsPanel()

        # ---- Controller ----
        self._controller = AnalysisController(parent=self)

        # ---- Layout ----
        self.setCentralWidget(self._plot_panel)
        self._setup_docks()
        self._setup_menu_bar()
        self._setup_status_bar()

        # ---- Wire signals ----
        self._connect_signals()

    # ==================================================================
    # Dock setup
    # ==================================================================

    def _setup_docks(self):
        # File panel — left
        self._dock_file = QDockWidget("File Browser", self)
        self._dock_file.setWidget(self._file_panel)
        self._dock_file.setMinimumWidth(220)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._dock_file)

        # Analysis panel — right
        self._dock_analysis = QDockWidget("Analysis", self)
        self._dock_analysis.setWidget(self._analysis_panel)
        self._dock_analysis.setMinimumWidth(260)
        self.addDockWidget(Qt.RightDockWidgetArea, self._dock_analysis)

        # Results panel — bottom
        self._dock_results = QDockWidget("Results", self)
        self._dock_results.setWidget(self._results_panel)
        self._dock_results.setMinimumHeight(160)
        self.addDockWidget(Qt.BottomDockWidgetArea, self._dock_results)

    # ==================================================================
    # Menu bar
    # ==================================================================

    def _setup_menu_bar(self):
        mb = self.menuBar()

        # ---- File ----
        file_menu = mb.addMenu("&File")
        act_open = file_menu.addAction("Open Folder…")
        act_open.triggered.connect(self._file_panel._on_open_folder)

        act_open_results = file_menu.addAction("Open Results…")
        act_open_results.triggered.connect(self._results_panel._open_results_file)

        file_menu.addSeparator()
        act_exit = file_menu.addAction("Exit")
        act_exit.triggered.connect(self.close)

        # ---- View ----
        view_menu = mb.addMenu("&View")
        view_menu.addAction(self._dock_file.toggleViewAction())
        view_menu.addAction(self._dock_analysis.toggleViewAction())
        view_menu.addAction(self._dock_results.toggleViewAction())

        view_menu.addSeparator()
        act_mpl = view_menu.addAction("Matplotlib backend")
        act_mpl.triggered.connect(lambda: self._plot_panel.set_backend("matplotlib"))
        act_pg = view_menu.addAction("PyQtGraph backend")
        act_pg.triggered.connect(lambda: self._plot_panel.set_backend("pyqtgraph"))

        # ---- Tools ----
        tools_menu = mb.addMenu("&Tools")
        # Post-hoc Analysis Wizard (if available)
        try:
            from pyAPisolation.gui.wizard_integration import add_analysis_wizard_to_menu
            add_analysis_wizard_to_menu(self, tools_menu)
        except Exception:
            pass

        # Prism Writer (if available)
        try:
            from pyAPisolation.dev.prism_writer_gui import PrismWriterGUI
            act_prism = tools_menu.addAction("Prism Writer")
            act_prism.triggered.connect(self._open_prism_writer)
        except ImportError:
            pass

        # Database Builder
        act_dbbuilder = tools_menu.addAction("Database Builder")
        act_dbbuilder.triggered.connect(self._open_database_builder)

    # ==================================================================
    # Status bar
    # ==================================================================

    def _setup_status_bar(self):
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    # ==================================================================
    # Signal wiring
    # ==================================================================

    def _connect_signals(self):
        # File panel → load + plot
        self._file_panel.file_selected.connect(self._on_file_selected)
        self._file_panel.sweeps_changed.connect(self._on_sweeps_changed)

        # Analysis panel → run
        self._analysis_panel.run_individual_requested.connect(self._on_run_individual)
        self._analysis_panel.run_batch_requested.connect(self._on_run_batch)

        # Plot panel span → analysis params
        self._plot_panel.span_selected.connect(self._on_span_selected)
        self._plot_panel.refresh_requested.connect(self._on_refresh)

        # Results panel → highlight file
        self._results_panel.file_highlight_requested.connect(
            self._file_panel.highlight_file
        )

        # Controller callbacks
        self._controller.individual_analysis_done.connect(self._on_individual_done)
        self._controller.batch_analysis_done.connect(self._on_batch_done)
        self._controller.analysis_error.connect(self._on_error)

    # ==================================================================
    # Slots
    # ==================================================================

    def _on_file_selected(self, name: str, path: str):
        """Load the selected ABF file, update sweep selector, and plot."""
        self._status.showMessage(f"Loading {name}…")
        try:
            cd = self._controller.load_file(path)
            self._file_panel.update_sweep_selector(cd.sweepCount)
            sweeps = self._file_panel.get_selected_sweeps()
            self._plot_file(cd, sweeps)
            self._status.showMessage(f"Loaded {name}  —  {cd.sweepCount} sweeps")
            # Auto-run individual analysis with current params
            self._on_run_individual()
        except Exception as exc:
            self._status.showMessage(f"Error loading {name}: {exc}")

    def _on_sweeps_changed(self, sweeps: list[int]):
        """Re-plot with updated sweep selection."""
        cd = self._controller.celldata
        if cd is not None:
            self._plot_file(cd, sweeps)

    def _on_run_individual(self):
        """Run analysis on the currently loaded file."""
        cd = self._controller.celldata
        if cd is None:
            return
        module_name = self._analysis_panel.get_selected_module_name()
        params = self._analysis_panel.get_params()
        sweeps = self._file_panel.get_selected_sweeps()
        show_rejected = self._analysis_panel.show_rejected()
        self._status.showMessage(f"Running {module_name}…")
        self._controller.run_individual(module_name, params, sweeps, show_rejected)

    def _on_run_batch(self):
        """Run batch analysis on the currently opened folder."""
        folder = self._file_panel._current_folder
        if not folder:
            QMessageBox.warning(self, "No Folder", "Open a folder first.")
            return
        module_name = self._analysis_panel.get_selected_module_name()
        params = self._analysis_panel.get_params()
        n_jobs = mp.cpu_count() if self._analysis_panel.is_parallel() else 1
        tag = self._analysis_panel.get_output_tag()
        proto = self._file_panel._combo_protocol.currentText()

        self._status.showMessage(f"Batch running {module_name}…")
        self._controller.run_batch(
            folder, module_name, params,
            protocol_filter=proto,
            n_jobs=n_jobs,
            output_tag=tag,
            parent_widget=self,
        )

    def _on_individual_done(self, result):
        """Handle completed individual analysis — update plot & results."""
        cd = self._controller.celldata
        sweeps = self._file_panel.get_selected_sweeps()
        dvdt_thr = self._analysis_panel.get_params().get("dv_cutoff")

        if isinstance(result, dict):
            # Legacy result dict
            spike_df = result.get("spike_df")
            rejected = result.get("rejected_spikes")
            subthres_df = result.get("subthres_df")
            self._plot_file(cd, sweeps, dvdt_threshold=dvdt_thr,
                            legacy_spike_df=spike_df,
                            legacy_rejected=rejected)
            # Update results table if we have data
            if spike_df:
                try:
                    combined = pd.concat(spike_df.values(), keys=spike_df.keys())
                    self._results_panel.set_dataframe(combined, index_col=None)
                except Exception:
                    pass
            elif subthres_df is not None:
                self._results_panel.set_dataframe(subthres_df, index_col=None)
        else:
            # Modular AnalysisResult
            rejected = getattr(result, "metadata", {}).get("rejected")
            #check if the result has a legacy_spike_df in its metadata, and if so use it for plotting
            legacy_spike_df = getattr(result, "metadata", {}).get("legacy_spike_df")
            self._plot_file(cd, sweeps, analysis_result=result,
                            dvdt_threshold=dvdt_thr,
                            legacy_spike_df=legacy_spike_df,
                            legacy_rejected=rejected)
            try:
                df = result.to_dataframe()
                if not df.empty:
                    self._results_panel.set_dataframe(df, index_col=None)
            except Exception:
                pass

        self._status.showMessage("Individual analysis complete")

    def _on_batch_done(self, result):
        """Handle completed batch analysis — update results table."""
        try:
            if isinstance(result, pd.DataFrame):
                self._results_panel.set_dataframe(result)
            else:
                df = result.to_dataframe()
                self._results_panel.set_dataframe(df)
        except Exception as exc:
            self._status.showMessage(f"Batch done but could not display: {exc}")
        self._status.showMessage("Batch analysis complete")

    def _on_span_selected(self, xmin: float, xmax: float):
        """Update start/end params in the analysis panel from span selection."""
        current_vals = self._analysis_panel.get_params()
        if "start" in current_vals:
            current_vals["start"] = round(xmin, 4)
        if "end" in current_vals:
            current_vals["end"] = round(xmax, 4)
        self._analysis_panel._param_form.set_values(current_vals)

    def _on_refresh(self):
        """Refresh the plot with current data."""
        cd = self._controller.celldata
        if cd is not None:
            sweeps = self._file_panel.get_selected_sweeps()
            self._plot_file(cd, sweeps)
            # Re-run individual analysis
            self._on_run_individual()

    def _on_error(self, msg: str):
        self._status.showMessage(f"Error: {msg[:120]}")
        QMessageBox.critical(self, "Analysis Error", msg)

    # ==================================================================
    # Plotting helpers
    # ==================================================================

    def _plot_file(
        self,
        celldata,
        sweeps,
        analysis_result=None,
        dvdt_threshold=None,
        legacy_spike_df=None,
        legacy_rejected=None,
    ):
        """Plot traces, optionally with analysis overlay."""
        # Use PlotPanel's high-level method for the base traces
        self._plot_panel.plot_file(
            celldata, sweeps,
            analysis_result=analysis_result,
            dvdt_threshold=dvdt_threshold,
        )

        # If we have legacy dicts, overlay manually
        if legacy_spike_df:
            bk = self._plot_panel.backend
            for sweep_idx, sdf in legacy_spike_df.items():
                if sdf is not None and not sdf.empty:
                    if "threshold_t" in sdf.columns and "threshold_v" in sdf.columns:
                        bk.mark_spikes(
                            sdf["threshold_t"].values,
                            sdf["threshold_v"].values,
                        )
                    elif "peak_t" in sdf.columns and "peak_v" in sdf.columns:
                        bk.mark_spikes(
                            sdf["peak_t"].values,
                            sdf["peak_v"].values,
                        )

        if legacy_rejected:
            bk = self._plot_panel.backend
            for sweep_idx, rdf in legacy_rejected.items():
                if rdf is not None and not rdf.empty:
                    if "threshold_t" in rdf.columns and "threshold_v" in rdf.columns:
                        bk.mark_rejected(
                            rdf["threshold_t"].values,
                            rdf["threshold_v"].values,
                        )

        self._plot_panel.backend.draw()

    # ==================================================================
    # Tool launchers
    # ==================================================================

    def _open_prism_writer(self):
        try:
            from pyAPisolation.dev.prism_writer_gui import PrismWriterGUI
            self._prism_win = PrismWriterGUI()
            self._prism_win.show()
        except Exception as exc:
            QMessageBox.warning(self, "Prism Writer", str(exc))

    def _open_database_builder(self):
        from .database_builder import DatabaseBuilderWindow
        self._dbbuilder_win = DatabaseBuilderWindow()
        self._dbbuilder_win.show()


# ======================================================================
# Entry point
# ======================================================================

def main():
    mp.freeze_support()
    app = QApplication(sys.argv)
    app.setApplicationName("pyAPisolation")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
