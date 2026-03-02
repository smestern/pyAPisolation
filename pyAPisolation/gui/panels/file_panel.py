"""
FilePanel — dock widget contents for browsing ABF files.

Provides a folder selector, protocol filter combo, file list, and a
SweepSelector that appears once a file is selected.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QGroupBox,
)

from ..widgets.sweep_selector import SweepSelector


class FilePanel(QWidget):
    """
    File browser panel: folder selection → protocol filter → ABF list →
    sweep selector.

    Signals
    -------
    folder_opened(str)
        Emitted when a folder is successfully opened.
    file_selected(str, str)
        ``(display_name, full_path)`` when a file is clicked.
    sweeps_changed(list)
        Forwarded from the embedded SweepSelector.
    protocol_changed(str)
        Emitted when the protocol filter changes.
    """

    folder_opened = Signal(str)
    file_selected = Signal(str, str)
    sweeps_changed = Signal(list)
    protocol_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._abf_pairs: list[tuple[str, str]] = []  # (name, path)
        self._protocol_file_map: dict[str, list[int]] = {}
        self._current_folder: str = ""
        self._setup_ui()

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Folder selection ---
        folder_row = QHBoxLayout()
        self._btn_open = QPushButton("Open Folder")
        self._btn_open.clicked.connect(self._on_open_folder)
        folder_row.addWidget(self._btn_open)
        self._lbl_folder = QLabel("No folder selected")
        self._lbl_folder.setWordWrap(True)
        self._lbl_folder.setStyleSheet("color: gray; font-size: 10px;")
        folder_row.addWidget(self._lbl_folder, stretch=1)
        layout.addLayout(folder_row)

        # --- Protocol filter ---
        proto_row = QHBoxLayout()
        proto_row.addWidget(QLabel("Protocol:"))
        self._combo_protocol = QComboBox()
        self._combo_protocol.addItem("[No Filter]")
        self._combo_protocol.currentTextChanged.connect(self._on_protocol_changed)
        proto_row.addWidget(self._combo_protocol, stretch=1)
        layout.addLayout(proto_row)

        # --- File list ---
        self._file_list = QListWidget()
        self._file_list.setAlternatingRowColors(True)
        self._file_list.currentRowChanged.connect(self._on_file_clicked)
        layout.addWidget(self._file_list, stretch=3)

        # --- Sweep selector ---
        sweep_group = QGroupBox("Sweeps")
        sweep_layout = QVBoxLayout(sweep_group)
        self._sweep_selector = SweepSelector()
        self._sweep_selector.selection_changed.connect(self.sweeps_changed.emit)
        sweep_layout.addWidget(self._sweep_selector)
        layout.addWidget(sweep_group, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_selected_sweeps(self) -> list[int]:
        return self._sweep_selector.get_selected()

    def get_current_file(self) -> Optional[tuple[str, str]]:
        row = self._file_list.currentRow()
        if 0 <= row < len(self._abf_pairs):
            return self._abf_pairs[row]
        return None

    def highlight_file(self, filename: str):
        """Select the list item whose display name matches *filename*."""
        for i in range(self._file_list.count()):
            if self._file_list.item(i).text() == filename:
                self._file_list.setCurrentRow(i)
                break

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select ABF folder")
        if not folder:
            return
        self._current_folder = folder
        self._lbl_folder.setText(folder)
        self._scan_folder(folder)
        self.folder_opened.emit(folder)

    def _scan_folder(self, folder: str):
        abf_files = sorted(glob.glob(os.path.join(folder, "**", "*.abf"), recursive=True))
        self._abf_pairs = [(os.path.basename(f), f) for f in abf_files]

        # Scan protocols
        progress = QProgressDialog("Scanning files…", "Cancel", 0, len(abf_files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)

        protocols: set[str] = set()
        self._protocol_file_map.clear()

        for idx, (name, path) in enumerate(self._abf_pairs):
            if progress.wasCanceled():
                break
            progress.setValue(idx)
            try:
                from pyAPisolation.dataset import cellData
                cd = cellData(path)
                proto = getattr(cd, "protocol", "")
                if proto:
                    protocols.add(proto)
                    self._protocol_file_map.setdefault(proto, []).append(idx)
            except Exception:
                pass
        progress.setValue(len(abf_files))

        # Populate protocol combo
        self._combo_protocol.blockSignals(True)
        self._combo_protocol.clear()
        self._combo_protocol.addItem("[No Filter]")
        for p in sorted(protocols):
            self._combo_protocol.addItem(p)
        self._combo_protocol.blockSignals(False)

        # Populate file list (unfiltered)
        self._populate_file_list()

    def _populate_file_list(self, indices=None):
        self._file_list.clear()
        if indices is None:
            indices = range(len(self._abf_pairs))
        for i in indices:
            self._file_list.addItem(self._abf_pairs[i][0])

    def _on_protocol_changed(self, text: str):
        if text == "[No Filter]" or text == "":
            self._populate_file_list()
        else:
            indices = self._protocol_file_map.get(text, [])
            self._populate_file_list(indices)
        self.protocol_changed.emit(text)

    def _on_file_clicked(self, row: int):
        if row < 0 or row >= self._file_list.count():
            return
        # Map the displayed name back to the full pair
        display_name = self._file_list.item(row).text()
        # Find matching pair
        for name, path in self._abf_pairs:
            if name == display_name:
                self.file_selected.emit(name, path)
                return

    # ------------------------------------------------------------------
    # Called externally after ABF is loaded
    # ------------------------------------------------------------------

    def update_sweep_selector(self, sweep_count: int):
        """Rebuild the sweep checkboxes for a newly-loaded file."""
        self._sweep_selector.set_sweeps(sweep_count, initially_checked=True)
