"""
PandasModel — QAbstractTableModel adapter for pandas DataFrames.

Extracted from the legacy spikeFinder.py so both old and new GUIs
can share the same model.
"""

import pandas as pd
from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt
from PySide6.QtGui import QColor


class PandasModel(QAbstractTableModel):
    """A model to interface a Qt view with a pandas DataFrame."""

    def __init__(self, dataframe: pd.DataFrame, index=None, parent=None):
        super().__init__(parent)
        if index is not None and index in dataframe.columns:
            dataframe = dataframe.copy()
            dataframe["_temp_index"] = dataframe[index].to_numpy()
            dataframe = dataframe.set_index("_temp_index")
        self._dataframe = dataframe

    # ------------------------------------------------------------------
    # Required overrides
    # ------------------------------------------------------------------

    def rowCount(self, parent=QModelIndex()) -> int:
        if parent == QModelIndex():
            return len(self._dataframe)
        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])

        if role == Qt.BackgroundRole and "outlier" in self._dataframe.columns:
            if self._dataframe.iloc[index.row()]["outlier"] == -1:
                return QColor(255, 200, 200)

        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns.values[section])
            if orientation == Qt.Vertical:
                return str(self._dataframe.index.values[section])
        return None

    def sort(self, column: int, order=Qt.SortOrder):
        self.layoutAboutToBeChanged.emit()
        self._dataframe = self._dataframe.sort_values(
            self._dataframe.columns[column],
            ascending=(order == Qt.AscendingOrder),
        )
        self.layoutChanged.emit()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def get_dataframe(self) -> pd.DataFrame:
        """Return the underlying DataFrame."""
        return self._dataframe

    def update_dataframe(self, df: pd.DataFrame, index=None):
        """Replace the entire DataFrame and refresh the view."""
        self.beginResetModel()
        if index is not None and index in df.columns:
            df = df.copy()
            df["_temp_index"] = df[index].to_numpy()
            df = df.set_index("_temp_index")
        self._dataframe = df
        self.endResetModel()
