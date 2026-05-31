"""
ResultsPanel — dock widget with a sortable QTableView backed by a
PandasModel, plus export buttons.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from ..widgets.pandas_model import PandasModel


class ResultsPanel(QWidget):
    """
    Sortable results table with CSV / Excel export.

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
        self._model: Optional[PandasModel] = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Table
        self._table = QTableView()
        self._table.setSortingEnabled(True)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setSelectionBehavior(QTableView.SelectRows)
        self._table.clicked.connect(self._on_row_clicked)
        layout.addWidget(self._table, stretch=1)

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
    # Public API
    # ------------------------------------------------------------------

    def set_dataframe(self, df: pd.DataFrame, index_col: str | None = "filename"):
        """Replace the table contents with a new DataFrame."""
        self._model = PandasModel(df, index=index_col, parent=self._table)
        self._table.setModel(self._model)

    def get_dataframe(self) -> Optional[pd.DataFrame]:
        if self._model is not None:
            return self._model.get_dataframe()
        return None

    def clear(self):
        self._table.setModel(None)
        self._model = None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_row_clicked(self, index):
        if self._model is None:
            return
        df = self._model.get_dataframe()
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
        if self._model is None:
            return
        df = self._model.get_dataframe()
        if fmt == "csv":
            path, _ = QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
            if path:
                df.to_csv(path, index=False)
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save Excel", "", "Excel Files (*.xlsx)")
            if path:
                df.to_excel(path, index=False)

    def _open_results_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Results", "",
            "Spreadsheets (*.csv *.xlsx *.xls)"
        )
        if not path:
            return
        if path.endswith(".csv"):
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path)
        self.set_dataframe(df)
