"""
AnalysisPanel — dock widget for selecting an analysis module, configuring
its parameters, and running individual / batch analysis.

The parameter form is auto-generated from ``AnalysisBase.get_parameters()``.
"""

from __future__ import annotations

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..widgets.param_form import ParamFormWidget


class AnalysisPanel(QWidget):
    """
    Lists registered analysis modules in a combo box, displays an
    auto-generated parameter form, and provides Run buttons.

    Signals
    -------
    module_changed(str)
        Emitted when the selected analysis module changes.
    params_changed(dict)
        Forwarded from the inner ParamFormWidget.
    run_individual_requested()
        "Run Current File" clicked.
    run_batch_requested()
        "Run Batch" clicked.
    """

    module_changed = Signal(str)
    params_changed = Signal(dict)
    run_individual_requested = Signal()
    run_batch_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._modules: dict[str, object] = {}  # name → AnalysisBase instance
        self._setup_ui()
        self._populate_modules()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Module selector ---
        mod_row = QHBoxLayout()
        mod_row.addWidget(QLabel("Analysis:"))
        self._combo_module = QComboBox()
        self._combo_module.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._combo_module.currentTextChanged.connect(self._on_module_changed)
        mod_row.addWidget(self._combo_module, stretch=1)
        layout.addLayout(mod_row)

        # --- Parameter form (scrollable) ---
        param_group = QGroupBox("Parameters")
        param_layout = QVBoxLayout(param_group)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._param_form = ParamFormWidget()
        self._param_form.params_changed.connect(self.params_changed.emit)
        self._scroll.setWidget(self._param_form)
        param_layout.addWidget(self._scroll)
        layout.addWidget(param_group, stretch=1)

        # --- Options ---
        opts_group = QGroupBox("Options")
        opts_layout = QVBoxLayout(opts_group)
        self._chk_parallel = QCheckBox("Enable Parallel Processing")
        opts_layout.addWidget(self._chk_parallel)
        self._chk_rejected = QCheckBox("Show Rejected Spikes")
        opts_layout.addWidget(self._chk_rejected)
        layout.addWidget(opts_group)

        # --- Output tag ---
        tag_row = QHBoxLayout()
        tag_row.addWidget(QLabel("Output tag:"))
        self._txt_tag = QLineEdit()
        self._txt_tag.setPlaceholderText("optional suffix for saved files")
        tag_row.addWidget(self._txt_tag, stretch=1)
        layout.addLayout(tag_row)

        # --- Run buttons ---
        btn_row = QHBoxLayout()
        self._btn_run_file = QPushButton("Run Current File")
        self._btn_run_file.clicked.connect(self.run_individual_requested.emit)
        btn_row.addWidget(self._btn_run_file)
        self._btn_run_batch = QPushButton("Run Batch")
        self._btn_run_batch.setStyleSheet("font-weight: bold;")
        self._btn_run_batch.clicked.connect(self.run_batch_requested.emit)
        btn_row.addWidget(self._btn_run_batch)
        layout.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Module population
    # ------------------------------------------------------------------

    def _populate_modules(self):
        """Fill the combo box from the analysis registry."""
        try:
            from pyAPisolation.analysis import get_all
            registry = get_all()
            for name, module_cls in registry.items():
                instance = module_cls() if isinstance(module_cls, type) else module_cls
                display = getattr(instance, "display_name", name)
                self._modules[display] = instance
                self._combo_module.addItem(display)
        except Exception as exc:
            # Analysis framework unavailable — add a placeholder
            self._combo_module.addItem("(no modules found)")
            print(f"[AnalysisPanel] Could not load analysis modules: {exc}")

    def refresh_modules(self):
        """Re-scan the registry (e.g. after a plugin is loaded at runtime)."""
        self._combo_module.blockSignals(True)
        self._combo_module.clear()
        self._modules.clear()
        self._populate_modules()
        self._combo_module.blockSignals(False)
        if self._combo_module.count():
            self._on_module_changed(self._combo_module.currentText())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_selected_module_name(self) -> str:
        """Return the *name* attribute of the selected module."""
        inst = self._get_current_module()
        return inst.name if inst else ""

    def get_selected_module(self):
        """Return the current AnalysisBase instance (or None)."""
        return self._get_current_module()

    def get_params(self) -> dict:
        """Return the current parameter values from the form."""
        return self._param_form.get_values()

    def is_parallel(self) -> bool:
        return self._chk_parallel.isChecked()

    def show_rejected(self) -> bool:
        return self._chk_rejected.isChecked()

    def get_output_tag(self) -> str:
        return self._txt_tag.text().strip()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_current_module(self):
        text = self._combo_module.currentText()
        return self._modules.get(text)

    def _on_module_changed(self, display_name: str):
        module = self._modules.get(display_name)
        if module is None:
            self._param_form.build({})
            return
        try:
            params = module.get_parameters()
            self._param_form.build(params)
        except Exception:
            self._param_form.build({})
        self.module_changed.emit(module.name)
