"""
ParamFormWidget — dynamically builds a QFormLayout from AnalysisBase parameter
metadata.  Supports float, int, bool, and str parameter types with sensible
spin-box ranges.
"""

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QWidget,
)


class ParamFormWidget(QWidget):
    """
    A self-building parameter form.

    Parameters
    ----------
    params : dict
        Output of ``AnalysisBase.get_parameters()``::

            {"dv_cutoff": {"type": float, "default": 7.0, "value": 7.0}, ...}
    """

    params_changed = Signal(dict)  # emits full {name: value} dict

    def __init__(self, params: dict | None = None, parent=None):
        super().__init__(parent)
        self._form = QFormLayout(self)
        self._form.setContentsMargins(4, 4, 4, 4)
        self._widgets: dict[str, QWidget] = {}
        if params:
            self.build(params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self, params: dict):
        """Tear down the current form and rebuild from *params*."""
        self._clear()
        for name, info in params.items():
            ptype = info.get("type", str)
            value = info.get("value", info.get("default"))
            widget = self._make_widget(name, ptype, value)
            # Pretty label: replace underscores, title-case
            label = name.replace("_", " ").title()
            self._form.addRow(label, widget)
            self._widgets[name] = widget

    def get_values(self) -> dict:
        """Return current parameter values as a plain dict."""
        values = {}
        for name, widget in self._widgets.items():
            if isinstance(widget, (QDoubleSpinBox, QSpinBox)):
                values[name] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[name] = widget.isChecked()
            elif isinstance(widget, QLineEdit):
                values[name] = widget.text()
        return values

    def set_values(self, values: dict):
        """Set parameter values from a dict (silently ignores unknown keys)."""
        for name, value in values.items():
            widget = self._widgets.get(name)
            if widget is None:
                continue
            if isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QLineEdit):
                widget.setText(str(value))

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _clear(self):
        """Remove all rows from the form."""
        while self._form.rowCount():
            self._form.removeRow(0)
        self._widgets.clear()

    def _make_widget(self, name: str, ptype, value) -> QWidget:
        if ptype is float or ptype == float:
            w = QDoubleSpinBox()
            w.setDecimals(4)
            w.setRange(-1e9, 1e9)
            w.setSingleStep(0.1)
            if value is not None:
                w.setValue(float(value))
            w.valueChanged.connect(self._on_changed)
            return w

        if ptype is int or ptype == int:
            w = QSpinBox()
            w.setRange(-999999, 999999)
            if value is not None:
                w.setValue(int(value))
            w.valueChanged.connect(self._on_changed)
            return w

        if ptype is bool or ptype == bool:
            w = QCheckBox()
            if value is not None:
                w.setChecked(bool(value))
            w.stateChanged.connect(self._on_changed)
            return w

        # Default: string
        w = QLineEdit()
        if value is not None:
            w.setText(str(value))
        w.editingFinished.connect(self._on_changed)
        return w

    def _on_changed(self, *_args):
        self.params_changed.emit(self.get_values())
