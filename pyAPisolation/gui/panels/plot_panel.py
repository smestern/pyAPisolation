"""
PlotPanel — central widget that wraps the switchable plot backends and
provides a high-level API for plotting electrophysiology traces.
"""

from __future__ import annotations

from typing import Optional, Callable

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..widgets.plot_backends import MatplotlibBackend, PyQtGraphBackend, PlotBackend


class PlotPanel(QWidget):
    """
    Central plotting area.  Contains a ``QStackedWidget`` toggling between
    matplotlib and pyqtgraph, plus a small toolbar with Refresh and
    backend-switch buttons.

    Signals
    -------
    refresh_requested()
        Emitted when the user clicks the Refresh button.
    span_selected(float, float)
        Emitted when the user drags a time-range span.
    backend_changed(str)
        ``"matplotlib"`` or ``"pyqtgraph"`` after a switch.
    """

    refresh_requested = Signal()
    span_selected = Signal(float, float)
    backend_changed = Signal(str)

    BACKEND_MPL = "matplotlib"
    BACKEND_PG = "pyqtgraph"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_backend_name: str = self.BACKEND_MPL
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Toolbar
        toolbar = QHBoxLayout()
        self._btn_refresh = QPushButton("Refresh")
        self._btn_refresh.setFixedWidth(80)
        self._btn_refresh.clicked.connect(self.refresh_requested.emit)
        toolbar.addWidget(self._btn_refresh)

        self._btn_toggle = QToolButton()
        self._btn_toggle.setText("Switch to PyQtGraph")
        self._btn_toggle.setCheckable(False)
        self._btn_toggle.clicked.connect(self._toggle_backend)
        toolbar.addWidget(self._btn_toggle)
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Stacked backends
        self._stack = QStackedWidget()
        self._mpl = MatplotlibBackend()
        self._stack.addWidget(self._mpl)  # index 0

        # PyQtGraph is optional — try to create it
        try:
            self._pg = PyQtGraphBackend()
            self._stack.addWidget(self._pg)  # index 1
            self._has_pg = True
        except ImportError:
            self._pg = None
            self._has_pg = False
            self._btn_toggle.setEnabled(False)
            self._btn_toggle.setToolTip("pyqtgraph not installed")

        layout.addWidget(self._stack, stretch=1)

        # Bind span callbacks
        self._mpl.set_span_callback(self._on_span)
        if self._has_pg:
            self._pg.set_span_callback(self._on_span)

    # ------------------------------------------------------------------
    # Backend switching
    # ------------------------------------------------------------------

    @property
    def backend(self) -> PlotBackend:
        """Return the currently active backend widget."""
        if self._current_backend_name == self.BACKEND_PG and self._has_pg:
            return self._pg
        return self._mpl

    def set_backend(self, name: str):
        """Switch to ``"matplotlib"`` or ``"pyqtgraph"``."""
        if name == self.BACKEND_PG and self._has_pg:
            self._stack.setCurrentWidget(self._pg)
            self._current_backend_name = self.BACKEND_PG
            self._btn_toggle.setText("Switch to Matplotlib")
        else:
            self._stack.setCurrentWidget(self._mpl)
            self._current_backend_name = self.BACKEND_MPL
            self._btn_toggle.setText("Switch to PyQtGraph")
        self.backend_changed.emit(self._current_backend_name)

    def _toggle_backend(self):
        if self._current_backend_name == self.BACKEND_MPL:
            self.set_backend(self.BACKEND_PG)
        else:
            self.set_backend(self.BACKEND_MPL)

    # ------------------------------------------------------------------
    # High-level plotting API
    # ------------------------------------------------------------------

    def clear(self):
        self.backend.clear()

    def plot_file(
        self,
        celldata,
        sweeps: list[int],
        analysis_result=None,
        rejected=None,
        dvdt_threshold: float | None = None,
    ):
        """
        Plot voltage traces and dV/dt for the given sweeps, optionally
        overlaying analysis results.

        Parameters
        ----------
        celldata
            A ``cellData`` object (or anything with ``setSweep`` /
            ``sweepX`` / ``sweepY``).
        sweeps
            List of sweep indices to plot.
        analysis_result : AnalysisResult, optional
            If provided, spike markers are drawn from sweep_results.
        rejected : dict, optional
            ``{sweep_idx: {"times": [...], "voltages": [...]}}`` for
            rejected-spike overlay.
        dvdt_threshold : float, optional
            If set, draw a horizontal dV/dt threshold line.
        """
        bk = self.backend
        bk.clear()

        # Color cycle (tab10)
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]

        for i, sweep_idx in enumerate(sweeps):
            try:
                celldata.setSweep(sweep_idx)
            except Exception:
                continue
            x = np.asarray(celldata.sweepX)
            y = np.asarray(celldata.sweepY)
            color = colors[i % len(colors)]

            bk.plot_sweep(x, y, color=color, label=f"Sweep {sweep_idx}")

            # Compute dV/dt
            if len(x) > 1:
                dt = np.diff(x)
                dvdt = np.diff(y) / dt / 1000.0  # mV/ms
                bk.plot_dvdt(x[:-1], dvdt, color=color)

        # Title
        title = getattr(celldata, "filePath", None) or ""
        if title:
            import os
            title = os.path.basename(title)
        bk.set_title(title)

        # dV/dt threshold line
        if dvdt_threshold is not None:
            bk.add_hline(dvdt_threshold, color="#FF0000", linestyle="--", axis="dvdt")

        # Spike markers from analysis result
        if analysis_result is not None:
            self._overlay_spikes(bk, analysis_result)

        # Rejected spikes
        if rejected is not None:
            self._overlay_rejected(bk, rejected)

        bk.draw()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _on_span(self, xmin: float, xmax: float):
        self.span_selected.emit(xmin, xmax)

    def _overlay_spikes(self, bk: PlotBackend, result):
        """Extract spike times/voltages from an AnalysisResult and mark them."""
        try:
            for sweep_res in result.sweep_results:
                times = sweep_res.get("threshold_t") or sweep_res.get("peak_t") or []
                volts = sweep_res.get("threshold_v") or sweep_res.get("peak_v") or []
                if not isinstance(times, (list, np.ndarray)):
                    times = [times]
                    volts = [volts]
                if times:
                    bk.mark_spikes(times, volts)
        except Exception:
            pass

    def _overlay_rejected(self, bk: PlotBackend, rejected: dict):
        """Overlay rejected spikes from the legacy ``determine_rejected_spikes`` output."""
        try:
            for sweep_idx, data in rejected.items():
                times = data.get("times", [])
                volts = data.get("voltages", [])
                if times:
                    bk.mark_rejected(times, volts)
        except Exception:
            pass
