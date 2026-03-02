"""
Plot backends — Matplotlib and PyQtGraph canvases behind a common API.

Both backends present two vertically stacked axes (voltage trace on top,
dV/dt on bottom) with linked X-axes and helpers for overlaying spike
markers and time-range selectors.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from PySide6.QtWidgets import QVBoxLayout, QWidget

if TYPE_CHECKING:
    pass

# ======================================================================
# Abstract interface
# ======================================================================

class PlotBackend(QWidget):
    """Minimal interface that all plot backends must satisfy."""

    def __init__(self, parent=None):
        super().__init__(parent)

    def clear(self):
        """Clear all plot contents."""
        raise NotImplementedError

    def plot_sweep(self, x, y, color: str = "#1f77b4", label: str = ""):
        """Plot a voltage trace on the primary axis."""
        raise NotImplementedError

    def plot_dvdt(self, x, dvdt, color: str = "#1f77b4"):
        """Plot dV/dt on the secondary axis."""
        raise NotImplementedError

    def mark_spikes(self, times, voltages, **kwargs):
        """Overlay spike markers on the primary axis."""
        raise NotImplementedError

    def mark_rejected(self, times, voltages, **kwargs):
        """Overlay rejected-spike markers on the primary axis."""
        raise NotImplementedError

    def add_hline(self, y: float, color: str = "#FF0000", linestyle: str = "--", axis: str = "dvdt"):
        """Draw a horizontal reference line on the specified axis."""
        raise NotImplementedError

    def set_span_callback(self, callback: Optional[Callable[[float, float], None]]):
        """Register a callback for interactive time-range selection."""
        raise NotImplementedError

    def set_title(self, title: str):
        """Set the plot title."""
        raise NotImplementedError

    def draw(self):
        """Flush / redraw the canvas."""
        raise NotImplementedError


# ======================================================================
# Matplotlib backend
# ======================================================================

class MatplotlibBackend(PlotBackend):
    """Dual-axis matplotlib canvas with NavigationToolbar and SpanSelector."""

    def __init__(self, parent=None):
        super().__init__(parent)
        # Lazy import so pyqtgraph-only users don't need matplotlib
        import matplotlib
        matplotlib.use("QtAgg")
        from matplotlib.backends.backend_qtagg import (
            FigureCanvasQTAgg as FigureCanvas,
            NavigationToolbar2QT as NavigationToolbar,
        )
        from matplotlib.figure import Figure
        from matplotlib.widgets import SpanSelector

        self._SpanSelector = SpanSelector

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._figure = Figure(tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._toolbar = NavigationToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        # Create linked axes
        self._ax_v = self._figure.add_subplot(211)
        self._ax_dv = self._figure.add_subplot(212, sharex=self._ax_v)

        self._span: Optional[SpanSelector] = None
        self._span_callback: Optional[Callable] = None

    # -- interface ---------------------------------------------------------

    def clear(self):
        self._ax_v.cla()
        self._ax_dv.cla()

    def plot_sweep(self, x, y, color="#1f77b4", label=""):
        self._ax_v.plot(x, y, color=color, label=label, linewidth=0.8)

    def plot_dvdt(self, x, dvdt, color="#1f77b4"):
        self._ax_dv.plot(x, dvdt, color=color, linewidth=0.8)

    def mark_spikes(self, times, voltages, **kwargs):
        color = kwargs.get("color", "#2ca02c")
        marker = kwargs.get("marker", "^")
        size = kwargs.get("size", 30)
        label = kwargs.get("label", "Spikes")
        self._ax_v.scatter(times, voltages, c=color, marker=marker,
                           s=size, zorder=5, label=label)

    def mark_rejected(self, times, voltages, **kwargs):
        color = kwargs.get("color", "#d62728")
        marker = kwargs.get("marker", "x")
        size = kwargs.get("size", 30)
        label = kwargs.get("label", "Rejected")
        self._ax_v.scatter(times, voltages, c=color, marker=marker,
                           s=size, zorder=5, label=label)

    def add_hline(self, y, color="#FF0000", linestyle="--", axis="dvdt"):
        ax = self._ax_dv if axis == "dvdt" else self._ax_v
        ax.axhline(y=y, color=color, ls=linestyle, linewidth=0.8)

    def set_span_callback(self, callback):
        self._span_callback = callback
        if callback is not None:
            self._span = self._SpanSelector(
                self._ax_v, self._on_span, "horizontal",
                useblit=True,
                props=dict(alpha=0.2, facecolor="tab:blue"),
                interactive=True,
                drag_from_anywhere=True,
            )
        else:
            self._span = None

    def set_title(self, title: str):
        self._ax_v.set_title(title, fontsize=10)

    def draw(self):
        self._ax_v.set_ylabel("Voltage (mV)")
        self._ax_dv.set_ylabel("dV/dt (mV/ms)")
        self._ax_dv.set_xlabel("Time (s)")
        if self._ax_v.get_legend_handles_labels()[1]:
            self._ax_v.legend(fontsize=7, loc="upper right")
        self._canvas.draw_idle()

    # -- internal ----------------------------------------------------------

    def _on_span(self, xmin, xmax):
        if self._span_callback:
            self._span_callback(xmin, xmax)


# ======================================================================
# PyQtGraph backend
# ======================================================================

class PyQtGraphBackend(PlotBackend):
    """Dual-axis pyqtgraph canvas with LinearRegionItem for span selection."""

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            import pyqtgraph as pg
            pg.setConfigOptions(background="w", foreground="k",
                                imageAxisOrder="row-major", useNumba=True)
        except ImportError:
            raise ImportError(
                "pyqtgraph is required for the PyQtGraph backend.  "
                "Install it with: pip install pyqtgraph"
            )

        self._pg = pg

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._glw = pg.GraphicsLayoutWidget()
        layout.addWidget(self._glw)

        self._plot_v = self._glw.addPlot(row=0, col=0)
        self._plot_dv = self._glw.addPlot(row=1, col=0)
        self._plot_dv.setXLink(self._plot_v)

        self._plot_v.addLegend(offset=(60, 10), labelTextSize="8pt")
        self._plot_v.setLabel("left", "Voltage", units="mV")
        self._plot_dv.setLabel("left", "dV/dt", units="mV/ms")
        self._plot_dv.setLabel("bottom", "Time", units="s")

        self._region: Optional[pg.LinearRegionItem] = None
        self._span_callback: Optional[Callable] = None

        # Track plot items for clearing
        self._sweep_items: list = []
        self._marker_items: list = []

    # -- interface ---------------------------------------------------------

    def clear(self):
        self._plot_v.clear()
        self._plot_dv.clear()
        self._sweep_items.clear()
        self._marker_items.clear()
        # Restore legend after clear
        self._plot_v.addLegend(offset=(60, 10), labelTextSize="8pt")
        # Restore region if it was active
        if self._region is not None and self._span_callback is not None:
            self._plot_v.addItem(self._region)

    def plot_sweep(self, x, y, color="#1f77b4", label=""):
        pen = self._pg.mkPen(color=color, width=1)
        item = self._plot_v.plot(x, y, pen=pen, name=label or None)
        self._sweep_items.append(item)

    def plot_dvdt(self, x, dvdt, color="#1f77b4"):
        pen = self._pg.mkPen(color=color, width=1)
        item = self._plot_dv.plot(x, dvdt, pen=pen)
        self._sweep_items.append(item)

    def mark_spikes(self, times, voltages, **kwargs):
        color = kwargs.get("color", "#2ca02c")
        symbol = kwargs.get("marker", "t")  # pyqtgraph triangle-up
        size = kwargs.get("size", 8)
        scatter = self._pg.ScatterPlotItem(
            x=np.asarray(times, dtype=float),
            y=np.asarray(voltages, dtype=float),
            pen=None, brush=color, symbol=symbol, size=size,
        )
        self._plot_v.addItem(scatter)
        self._marker_items.append(scatter)

    def mark_rejected(self, times, voltages, **kwargs):
        color = kwargs.get("color", "#d62728")
        symbol = kwargs.get("marker", "x")
        size = kwargs.get("size", 8)
        scatter = self._pg.ScatterPlotItem(
            x=np.asarray(times, dtype=float),
            y=np.asarray(voltages, dtype=float),
            pen=None, brush=color, symbol=symbol, size=size,
        )
        self._plot_v.addItem(scatter)
        self._marker_items.append(scatter)

    def add_hline(self, y, color="#FF0000", linestyle="--", axis="dvdt"):
        plot = self._plot_dv if axis == "dvdt" else self._plot_v
        pen = self._pg.mkPen(color=color, width=1,
                              style=self._pg.QtCore.Qt.DashLine)
        line = self._pg.InfiniteLine(pos=y, angle=0, pen=pen)
        plot.addItem(line)

    def set_span_callback(self, callback):
        self._span_callback = callback
        if callback is not None:
            if self._region is None:
                self._region = self._pg.LinearRegionItem(
                    values=[0, 1],
                    brush=self._pg.mkBrush(100, 100, 200, 40),
                )
                self._region.sigRegionChangeFinished.connect(self._on_region)
                self._plot_v.addItem(self._region)
        else:
            if self._region is not None:
                self._plot_v.removeItem(self._region)
                self._region = None

    def set_title(self, title: str):
        self._plot_v.setTitle(title, size="10pt")

    def draw(self):
        # pyqtgraph redraws automatically; nothing to flush
        pass

    # -- internal ----------------------------------------------------------

    def _on_region(self):
        if self._region and self._span_callback:
            rng = self._region.getRegion()
            self._span_callback(rng[0], rng[1])
