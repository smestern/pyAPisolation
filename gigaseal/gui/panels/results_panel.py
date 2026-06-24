"""
ResultsPanel — dock widget showing analysis results as one or more
sortable tables (one tab per sheet) backed by PandasModels, plus
export buttons.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, Optional

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableView,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..widgets.pandas_model import PandasModel


class ResultsPanel(QWidget):
    """
    Tabbed results view with CSV / Excel export.

    Each named sheet (e.g. ``"Summary"``, ``"Raw"``) is shown in its own
    tab as a sortable table.  A single DataFrame is shown as one tab.

    Signals
    -------
    file_highlight_requested(str)
        Emitted when the user clicks a row whose DataFrame contains a
        ``filename`` column — carries the filename string so the
        FilePanel can highlight it.
    """

    file_highlight_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Per-tab state, kept index-aligned with the tab widget.
        self._tables: list[QTableView] = []
        self._models: list[PandasModel] = []
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Tabbed tables
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        layout.addWidget(self._tabs, stretch=1)

        # Export buttons
        btn_row = QHBoxLayout()
        self._btn_csv = QPushButton("Export CSV")
        self._btn_csv.clicked.connect(lambda: self._export("csv"))
        btn_row.addWidget(self._btn_csv)
        self._btn_xlsx = QPushButton("Export Excel")
        self._btn_xlsx.clicked.connect(lambda: self._export("xlsx"))
        btn_row.addWidget(self._btn_xlsx)

        # Open results file
        self._btn_open = QPushButton("Open Results…")
        self._btn_open.clicked.connect(self._open_results_file)
        btn_row.addWidget(self._btn_open)

        btn_row.addStretch()
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Tab construction
    # ------------------------------------------------------------------

    def _make_table(self, df: pd.DataFrame, index_col: str | None) -> QTableView:
        """Build a sortable QTableView backed by a PandasModel for *df*."""
        table = QTableView()
        table.setSortingEnabled(True)
        table.setAlternatingRowColors(True)
        table.horizontalHeader().setStretchLastSection(True)
        table.setSelectionBehavior(QTableView.SelectRows)
        model = PandasModel(df, index=index_col, parent=table)
        table.setModel(model)
        table.clicked.connect(lambda index, t=table: self._on_row_clicked(t, index))
        self._tables.append(table)
        self._models.append(model)
        return table

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_sheets(
        self,
        sheets: Dict[str, pd.DataFrame],
        index_col: str | None = "filename",
    ):
        """
        Replace the view with one tab per named sheet.

        Sheets are shown in the order given (use an ``OrderedDict`` to
        control tab order).  Empty sheets are skipped; if every sheet is
        empty the view is cleared.
        """
        self.clear()
        for name, df in sheets.items():
            if df is None or df.empty:
                continue
            table = self._make_table(df, index_col)
            self._tabs.addTab(table, str(name))

    def set_dataframe(self, df: pd.DataFrame, index_col: str | None = "filename"):
        """Replace the view with a single-tab table (back-compat shim)."""
        self.set_sheets(OrderedDict([("Results", df)]), index_col=index_col)

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        """Return the DataFrame of the currently active tab, if any."""
        model = self._active_model()
        if model is not None:
            return model.get_dataframe()
        return None

    def get_sheets(self) -> "OrderedDict[str, pd.DataFrame]":
        """Return every tab's DataFrame keyed by its tab label."""
        out: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        for i in range(self._tabs.count()):
            out[self._tabs.tabText(i)] = self._models[i].get_dataframe()
        return out

    def clear(self):
        self._tabs.clear()
        self._tables.clear()
        self._models.clear()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _active_model(self) -> Optional[PandasModel]:
        idx = self._tabs.currentIndex()
        if 0 <= idx < len(self._models):
            return self._models[idx]
        return None

    def _on_row_clicked(self, table: QTableView, index):
        model = table.model()
        if model is None:
            return
        df = model.get_dataframe()
        try:
            row = df.iloc[index.row()]
            if "filename" in df.columns:
                fname = str(row["filename"])
                if not fname.endswith(".abf"):
                    fname += ".abf"
                self.file_highlight_requested.emit(fname)
        except Exception:
            pass

    def _export(self, fmt: str):
        if fmt == "csv":
            model = self._active_model()
            if model is None:
                return
            df = model.get_dataframe()
            path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if path:
                df.to_csv(path, index=False)
        else:
            sheets = self.get_sheets()
            if not sheets:
                return
            path, _ = QFileDialog.getSaveFileName(self, "Save Excel", "", "Excel Files (*.xlsx)")
            if path:
                with pd.ExcelWriter(path) as writer:
                    for name, df in sheets.items():
                        df.to_excel(writer, sheet_name=str(name)[:31], index=False)

    def _open_results_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Results", "",
            "Spreadsheets (*.csv *.xlsx *.xls)"
        )
        if not path:
            return
        if path.endswith(".csv"):
            self.set_dataframe(pd.read_csv(path))
        else:
            # Load every sheet so multi-sheet workbooks round-trip into tabs.
            book = pd.read_excel(path, sheet_name=None)
            self.set_sheets(OrderedDict(book))
