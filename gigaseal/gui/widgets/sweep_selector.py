"""
SweepSelector — a compact grid of checkboxes, one per sweep, with
Check All / Uncheck All toggle.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class SweepSelector(QWidget):
    """
    Displays a checkbox for each sweep and emits the list of selected
    sweep indices whenever the selection changes.
    """

    selection_changed = Signal(list)  # list[int]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._checkboxes: list[QCheckBox] = []
        self._outer = QVBoxLayout(self)
        self._outer.setContentsMargins(2, 2, 2, 2)
        self._grid = QGridLayout()
        self._outer.addLayout(self._grid)

        # Toggle button
        btn_row = QHBoxLayout()
        self._toggle_btn = QPushButton("Check All")
        self._toggle_btn.setFixedHeight(24)
        self._toggle_btn.clicked.connect(self._toggle_all)
        btn_row.addWidget(self._toggle_btn)
        btn_row.addStretch()
        self._outer.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_sweeps(self, sweep_count: int, initially_checked: bool = True):
        """Rebuild checkboxes for *sweep_count* sweeps."""
        self._clear_checkboxes()
        cols = max(4, int(sweep_count ** 0.5) + 1)
        for i in range(sweep_count):
            cb = QCheckBox(f"{i}")
            cb.setChecked(initially_checked)
            cb.stateChanged.connect(self._on_changed)
            self._grid.addWidget(cb, i // cols, i % cols)
            self._checkboxes.append(cb)
        self._update_toggle_label()

    def get_selected(self) -> list[int]:
        """Return indices of checked sweeps."""
        return [i for i, cb in enumerate(self._checkboxes) if cb.isChecked()]

    def set_selected(self, indices: list[int]):
        """Check only the given sweep indices."""
        for i, cb in enumerate(self._checkboxes):
            cb.blockSignals(True)
            cb.setChecked(i in indices)
            cb.blockSignals(False)
        self._on_changed()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _clear_checkboxes(self):
        for cb in self._checkboxes:
            self._grid.removeWidget(cb)
            cb.deleteLater()
        self._checkboxes.clear()

    def _toggle_all(self):
        any_unchecked = any(not cb.isChecked() for cb in self._checkboxes)
        for cb in self._checkboxes:
            cb.blockSignals(True)
            cb.setChecked(any_unchecked)
            cb.blockSignals(False)
        self._update_toggle_label()
        self._on_changed()

    def _update_toggle_label(self):
        if all(cb.isChecked() for cb in self._checkboxes):
            self._toggle_btn.setText("Uncheck All")
        else:
            self._toggle_btn.setText("Check All")

    def _on_changed(self, *_args):
        self._update_toggle_label()
        self.selection_changed.emit(self.get_selected())
