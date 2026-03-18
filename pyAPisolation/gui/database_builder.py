"""
Database Builder — standalone window for mapping recordings to cells.

Launch from the Tools menu or directly::

    python -m pyAPisolation.gui.database_builder
"""

from __future__ import annotations

import glob
import json
import os
import sys
from typing import Optional

from PySide6.QtCore import Qt, QMimeData, Signal
from PySide6.QtGui import QAction, QDragEnterEvent, QDropEvent, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDateEdit,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..database.tsDatabase import tsDatabase, CONDITION_SEP


# ======================================================================
# File tree (left panel) — drag source
# ======================================================================

class _FileTree(QTreeWidget):
    """Tree widget showing scanned recording files, drag-enabled."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["File", "Protocol"])
        self.setColumnCount(2)
        self.setDragEnabled(True)
        self.setSelectionMode(self.ExtendedSelection)
        self.setAlternatingRowColors(True)
        header = self.header()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(0, QHeaderView.Stretch)

    # --- drag support ---------------------------------------------------

    def mimeData(self, items):
        """Pack file path + protocol into JSON mime data."""
        data = QMimeData()
        entries = []
        for item in items:
            path = item.data(0, Qt.UserRole)
            proto = item.text(1)
            if path:
                entries.append({"path": path, "protocol": proto or ""})
        data.setData("application/x-dbbuilder-files", json.dumps(entries).encode())
        return data


# ======================================================================
# Cell-Protocol table (center panel) — drop target
# ======================================================================

class _CellTable(QTableWidget):
    """Table showing cell rows × protocol columns with drop support."""

    file_dropped = Signal(int, int, str, str)  # row, col, path, protocol

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropOverwriteMode(False)
        self.setAlternatingRowColors(True)
        self.setSelectionBehavior(self.SelectItems)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)

    # --- drop support ---------------------------------------------------

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasFormat("application/x-dbbuilder-files"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasFormat("application/x-dbbuilder-files"):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        raw = event.mimeData().data("application/x-dbbuilder-files")
        entries = json.loads(bytes(raw).decode())
        if not entries:
            return

        idx = self.indexAt(event.position().toPoint())
        row = idx.row()
        col = idx.column()
        if row < 0 or col < 0:
            event.ignore()
            return

        # Use first entry's data
        entry = entries[0]
        self.file_dropped.emit(row, col, entry["path"], entry["protocol"])
        event.acceptProposedAction()

    # --- context menu ---------------------------------------------------

    def _context_menu(self, pos):
        item = self.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        act_clear = menu.addAction("Unassign file")
        act_path = menu.addAction("Copy file path")
        chosen = menu.exec(self.viewport().mapToGlobal(pos))
        if chosen == act_clear:
            row, col = self.row(item), self.column(item)
            self.setItem(row, col, QTableWidgetItem(""))
            self.file_dropped.emit(row, col, "", "")
        elif chosen == act_path:
            tip = item.toolTip()
            if tip:
                QApplication.clipboard().setText(tip)


# ======================================================================
# DatabaseBuilderWindow
# ======================================================================

class DatabaseBuilderWindow(QMainWindow):
    """Standalone window for building a cell ↔ recording database."""

    TITLE = "Database Builder"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(self.TITLE)
        self.resize(1200, 700)

        self._db = tsDatabase()
        self._file_meta: dict[str, str] = {}  # path → protocol (cache)

        # ---- layout ----
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        self.setCentralWidget(splitter)

        self._build_menu_bar()
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready — create or open a database")

        # ---- signals ----
        self._cell_table.file_dropped.connect(self._on_file_dropped)
        self._cell_table.currentCellChanged.connect(self._on_cell_selected)

    # ==================================================================
    # Left panel — file browser
    # ==================================================================

    def _build_left_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        btn_row = QHBoxLayout()
        btn_open = QPushButton("Open Folder…")
        btn_open.clicked.connect(self._on_open_folder)
        btn_row.addWidget(btn_open)
        lay.addLayout(btn_row)

        self._lbl_folder = QLabel("No folder selected")
        self._lbl_folder.setStyleSheet("color: gray; font-size: 10px;")
        self._lbl_folder.setWordWrap(True)
        lay.addWidget(self._lbl_folder)

        # Filter
        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel("Filter:"))
        self._txt_filter = QLineEdit()
        self._txt_filter.setPlaceholderText("Type to filter files…")
        self._txt_filter.textChanged.connect(self._apply_file_filter)
        filt_row.addWidget(self._txt_filter)
        lay.addLayout(filt_row)

        self._file_tree = _FileTree()
        lay.addWidget(self._file_tree, stretch=1)
        return w

    # ==================================================================
    # Center panel — cell-protocol table
    # ==================================================================

    def _build_center_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(QLabel("Cell – Protocol Table"))

        self._cell_table = _CellTable()
        lay.addWidget(self._cell_table, stretch=1)

        # Quick-action buttons
        btn_row = QHBoxLayout()
        btn_add_cell = QPushButton("+ Cell")
        btn_add_cell.clicked.connect(self._on_add_cell)
        btn_row.addWidget(btn_add_cell)
        btn_rm_cell = QPushButton("- Cell")
        btn_rm_cell.clicked.connect(self._on_remove_cell)
        btn_row.addWidget(btn_rm_cell)
        btn_add_proto = QPushButton("+ Protocol")
        btn_add_proto.clicked.connect(self._on_add_protocol)
        btn_row.addWidget(btn_add_proto)
        btn_rm_proto = QPushButton("- Protocol")
        btn_rm_proto.clicked.connect(self._on_remove_protocol)
        btn_row.addWidget(btn_rm_proto)
        lay.addLayout(btn_row)
        return w

    # ==================================================================
    # Right panel — cell detail / metadata
    # ==================================================================

    def _build_right_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 4, 4)

        grp = QGroupBox("Cell Metadata")
        form = QFormLayout(grp)

        self._meta_name = QLineEdit()
        self._meta_name.setReadOnly(True)
        form.addRow("Cell:", self._meta_name)

        self._meta_condition = QComboBox()
        self._meta_condition.setEditable(True)
        self._meta_condition.setInsertPolicy(QComboBox.InsertAtBottom)
        form.addRow("Condition:", self._meta_condition)

        self._meta_group = QComboBox()
        self._meta_group.setEditable(True)
        self._meta_group.setInsertPolicy(QComboBox.InsertAtBottom)
        form.addRow("Group:", self._meta_group)

        self._meta_drug = QLineEdit()
        form.addRow("Drug:", self._meta_drug)

        self._meta_experimenter = QLineEdit()
        form.addRow("Experimenter:", self._meta_experimenter)

        self._meta_date = QLineEdit()
        form.addRow("Date:", self._meta_date)

        self._meta_notes = QTextEdit()
        self._meta_notes.setMaximumHeight(80)
        form.addRow("Notes:", self._meta_notes)

        lay.addWidget(grp)

        btn_apply = QPushButton("Apply Metadata")
        btn_apply.clicked.connect(self._apply_metadata)
        lay.addWidget(btn_apply)

        btn_dup = QPushButton("Duplicate Cell (clear files)")
        btn_dup.clicked.connect(self._on_duplicate_cell)
        lay.addWidget(btn_dup)

        # File summary
        self._lbl_file_summary = QLabel("")
        self._lbl_file_summary.setWordWrap(True)
        lay.addWidget(self._lbl_file_summary)

        lay.addStretch()
        return w

    # ==================================================================
    # Menu bar
    # ==================================================================

    def _build_menu_bar(self):
        mb = self.menuBar()

        # ---- File ----
        file_menu = mb.addMenu("&File")
        act_new = file_menu.addAction("New Database")
        act_new.setShortcut(QKeySequence.New)
        act_new.triggered.connect(self._on_new_db)

        act_open = file_menu.addAction("Open Database…")
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self._on_open_db)

        file_menu.addSeparator()
        act_save = file_menu.addAction("Save")
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self._on_save)

        act_save_as = file_menu.addAction("Save As…")
        act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self._on_save_as)

        file_menu.addSeparator()

        act_export_csv = file_menu.addAction("Export CSV…")
        act_export_csv.triggered.connect(self._on_export_csv)

        act_import_csv = file_menu.addAction("Import CSV…")
        act_import_csv.triggered.connect(self._on_import_csv)

        file_menu.addSeparator()

        act_open_folder = file_menu.addAction("Open Recording Folder…")
        act_open_folder.triggered.connect(self._on_open_folder)

        # ---- Edit ----
        edit_menu = mb.addMenu("&Edit")
        act_add_cell = edit_menu.addAction("Add Cell")
        act_add_cell.triggered.connect(self._on_add_cell)
        act_rm_cell = edit_menu.addAction("Remove Selected Cell")
        act_rm_cell.triggered.connect(self._on_remove_cell)
        edit_menu.addSeparator()
        act_add_proto = edit_menu.addAction("Add Protocol Column")
        act_add_proto.triggered.connect(self._on_add_protocol)
        act_rm_proto = edit_menu.addAction("Remove Selected Protocol")
        act_rm_proto.triggered.connect(self._on_remove_protocol)
        edit_menu.addSeparator()
        act_rename_cell = edit_menu.addAction("Rename Cell")
        act_rename_cell.triggered.connect(self._on_rename_cell)

    # ==================================================================
    # Folder scanning
    # ==================================================================

    def _on_open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select recording folder")
        if not folder:
            return
        self._lbl_folder.setText(folder)
        self._db.path = folder
        self._scan_folder(folder)

    def _scan_folder(self, folder: str):
        patterns = ["**/*.abf", "**/*.nwb"]
        files = []
        for pat in patterns:
            files.extend(glob.glob(os.path.join(folder, pat), recursive=True))
        files = sorted(set(files))

        progress = QProgressDialog("Scanning files…", "Cancel", 0, len(files), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(300)

        self._file_tree.clear()
        self._file_meta.clear()
        folder_items: dict[str, QTreeWidgetItem] = {}

        for idx, fpath in enumerate(files):
            if progress.wasCanceled():
                break
            progress.setValue(idx)

            rel = os.path.relpath(fpath, folder)
            parts = rel.split(os.sep)
            fname = parts[-1]
            subfolder = os.sep.join(parts[:-1]) if len(parts) > 1 else ""

            # Detect protocol
            proto = ""
            try:
                from ..dataset import cellData
                cd = cellData(fpath)
                proto = getattr(cd, "protocol", "") or ""
            except Exception:
                pass
            self._file_meta[fpath] = proto

            # Get or create parent folder item
            if subfolder and subfolder not in folder_items:
                parent = QTreeWidgetItem(self._file_tree, [subfolder, ""])
                parent.setFlags(parent.flags() & ~Qt.ItemIsDragEnabled)
                folder_items[subfolder] = parent
            parent_item = folder_items.get(subfolder)

            if parent_item:
                item = QTreeWidgetItem(parent_item, [fname, proto])
            else:
                item = QTreeWidgetItem(self._file_tree, [fname, proto])

            item.setData(0, Qt.UserRole, fpath)
            item.setToolTip(0, fpath)

        progress.setValue(len(files))
        self._file_tree.expandAll()
        self._status.showMessage(f"Scanned {len(files)} files in {folder}")

    def _apply_file_filter(self, text: str):
        text_lower = text.lower()
        root = self._file_tree.invisibleRootItem()
        self._filter_tree_item(root, text_lower)

    def _filter_tree_item(self, item: QTreeWidgetItem, text: str) -> bool:
        """Recursively show/hide items matching *text*. Returns True if visible."""
        if item.childCount() == 0:
            visible = text in item.text(0).lower() or text in item.text(1).lower()
            item.setHidden(not visible)
            return visible
        any_visible = False
        for i in range(item.childCount()):
            if self._filter_tree_item(item.child(i), text):
                any_visible = True
        item.setHidden(not any_visible)
        return any_visible

    # ==================================================================
    # Cell CRUD
    # ==================================================================

    def _on_add_cell(self):
        name = self._db.next_cell_name()
        self._db.add_cell(name)
        self._refresh_table()
        self._status.showMessage(f"Added cell {name}")

    def _on_remove_cell(self):
        row = self._cell_table.currentRow()
        if row < 0:
            return
        name = self._db.cell_names()[row]
        self._db.remove_cell(name)
        self._refresh_table()
        self._status.showMessage(f"Removed cell {name}")

    def _on_rename_cell(self):
        row = self._cell_table.currentRow()
        if row < 0:
            return
        old = self._db.cell_names()[row]
        new, ok = QInputDialog.getText(self, "Rename Cell", "New name:", text=old)
        if ok and new and new != old:
            self._db.rename_cell(old, new)
            self._refresh_table()

    def _on_duplicate_cell(self):
        row = self._cell_table.currentRow()
        if row < 0:
            return
        src = self._db.cell_names()[row]
        new_name = self._db.next_cell_name()
        meta = {}
        for col in self._db.get_metadata_columns():
            val = self._db.cellindex.loc[src, col]
            if val is not None:
                meta[col] = val
        self._db.add_cell(new_name, metadata=meta)
        self._refresh_table()
        self._status.showMessage(f"Duplicated {src} → {new_name} (files cleared)")

    # ==================================================================
    # Protocol CRUD
    # ==================================================================

    def _on_add_protocol(self):
        name, ok = QInputDialog.getText(self, "Add Protocol", "Protocol name:")
        if not ok or not name:
            return
        # Ask about condition
        cond, ok2 = QInputDialog.getText(
            self, "Condition (optional)",
            "Condition label (leave blank for none):"
        )
        cond = cond.strip() if ok2 and cond else None
        self._db.add_protocol(name.strip(), condition=cond)
        self._refresh_table()
        self._status.showMessage(f"Added protocol column: {self._db._col_name(name.strip(), cond)}")

    def _on_remove_protocol(self):
        col = self._cell_table.currentColumn()
        if col < 0:
            return
        proto_cols = self._db.get_protocol_columns()
        if col >= len(proto_cols):
            return
        col_name = proto_cols[col]
        base = self._db.protocol_base_name(col_name)
        cond = self._db.protocol_condition(col_name)
        self._db.remove_protocol(base, condition=cond)
        self._refresh_table()
        self._status.showMessage(f"Removed protocol column: {col_name}")

    # ==================================================================
    # Drag-and-drop handling
    # ==================================================================

    def _on_file_dropped(self, row: int, col: int, path: str, drag_proto: str):
        cells = self._db.cell_names()
        proto_cols = self._db.get_protocol_columns()
        if row >= len(cells) or col >= len(proto_cols):
            return

        cell_name = cells[row]
        col_name = proto_cols[col]

        if not path:
            # Unassign
            base = self._db.protocol_base_name(col_name)
            cond = self._db.protocol_condition(col_name)
            self._db.unassign_file(cell_name, base, condition=cond)
            self._refresh_table()
            return

        # If dragged protocol doesn't match column, offer options
        base = self._db.protocol_base_name(col_name)
        cond = self._db.protocol_condition(col_name)

        # Auto-create column if protocol from file doesn't exist yet
        if drag_proto and drag_proto != base:
            # Offer to create new column or use target column
            resp = QMessageBox.question(
                self, "Protocol Mismatch",
                f"The file's protocol is '{drag_proto}' but you dropped it on "
                f"'{col_name}'.\n\n"
                f"Create a new '{drag_proto}' column instead?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
            )
            if resp == QMessageBox.Cancel:
                return
            if resp == QMessageBox.Yes:
                # Check if we need a condition for the new column
                self._db.add_protocol(drag_proto)
                self._db.assign_file(cell_name, drag_proto, path)
                self._refresh_table()
                self._status.showMessage(
                    f"Assigned {os.path.basename(path)} → {cell_name} / {drag_proto}"
                )
                return

        self._db.assign_file(cell_name, base, path, condition=cond)
        self._refresh_table()
        self._status.showMessage(
            f"Assigned {os.path.basename(path)} → {cell_name} / {col_name}"
        )

    # ==================================================================
    # Table refresh
    # ==================================================================

    def _refresh_table(self):
        """Rebuild cell table from the database model."""
        self._cell_table.blockSignals(True)
        cells = self._db.cell_names()
        proto_cols = self._db.get_protocol_columns()

        self._cell_table.setRowCount(len(cells))
        self._cell_table.setColumnCount(len(proto_cols))
        self._cell_table.setHorizontalHeaderLabels(proto_cols)
        self._cell_table.setVerticalHeaderLabels(cells)

        for r, cell in enumerate(cells):
            for c, col in enumerate(proto_cols):
                val = self._db.cellindex.loc[cell, col]
                if val is not None and str(val).strip() and str(val) != "nan":
                    paths = str(val).split(";")
                    display = ", ".join(os.path.basename(p) for p in paths if p.strip())
                    item = QTableWidgetItem(display)
                    item.setToolTip(str(val))
                else:
                    item = QTableWidgetItem("")
                item.setFlags(item.flags() | Qt.ItemIsDropEnabled)
                self._cell_table.setItem(r, c, item)

        self._cell_table.blockSignals(False)

    # ==================================================================
    # Cell metadata panel
    # ==================================================================

    def _on_cell_selected(self, row, col, prev_row, prev_col):
        cells = self._db.cell_names()
        if row < 0 or row >= len(cells):
            self._meta_name.clear()
            return
        cell = cells[row]
        data = self._db.get_cell(cell)

        self._meta_name.setText(cell)
        self._set_combo(self._meta_condition, str(data.get("condition", "") or ""))
        self._set_combo(self._meta_group, str(data.get("group", "") or ""))
        self._meta_drug.setText(str(data.get("drug", "") or ""))
        self._meta_experimenter.setText(str(data.get("experimenter", "") or ""))
        self._meta_date.setText(str(data.get("date", "") or ""))
        self._meta_notes.setPlainText(str(data.get("notes", "") or ""))

        # file summary
        proto_cols = self._db.get_protocol_columns()
        assigned = sum(1 for c in proto_cols
                       if c in data and data[c] is not None and str(data[c]).strip() and str(data[c]) != "nan")
        self._lbl_file_summary.setText(
            f"{assigned} / {len(proto_cols)} protocol columns filled"
        )

    def _set_combo(self, combo: QComboBox, value: str):
        """Set combo value, adding it if it's not already there."""
        idx = combo.findText(value)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        else:
            if value and value != "nan":
                combo.addItem(value)
                combo.setCurrentText(value)
            else:
                combo.setCurrentText("")

    def _apply_metadata(self):
        cell = self._meta_name.text()
        if not cell or cell not in self._db.cellindex.index:
            return
        self._db.set_cell_metadata(cell, "condition", self._meta_condition.currentText())
        self._db.set_cell_metadata(cell, "group", self._meta_group.currentText())
        self._db.set_cell_metadata(cell, "drug", self._meta_drug.text())
        self._db.set_cell_metadata(cell, "experimenter", self._meta_experimenter.text())
        self._db.set_cell_metadata(cell, "date", self._meta_date.text())
        self._db.set_cell_metadata(cell, "notes", self._meta_notes.toPlainText())

        # Update condition/group combos globally
        self._update_combo_options()
        self._status.showMessage(f"Metadata updated for {cell}")

    def _update_combo_options(self):
        """Refresh combo options from all cells' metadata."""
        for combo, key in [(self._meta_condition, "condition"), (self._meta_group, "group")]:
            current = combo.currentText()
            vals = set()
            if key in self._db.cellindex.columns:
                for v in self._db.cellindex[key].dropna().unique():
                    s = str(v).strip()
                    if s and s != "nan":
                        vals.add(s)
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(sorted(vals))
            self._set_combo(combo, current)
            combo.blockSignals(False)

    # ==================================================================
    # File operations
    # ==================================================================

    def _on_new_db(self):
        if self._db.cell_count() > 0:
            if QMessageBox.question(
                self, "New Database",
                "Discard current database?",
                QMessageBox.Yes | QMessageBox.No,
            ) != QMessageBox.Yes:
                return
        self._db.clear()
        self._refresh_table()
        self._status.showMessage("New database created")

    def _on_open_db(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Database", "",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        if not path:
            return
        self._db.load_xlsx(path)
        self._refresh_table()
        self._update_combo_options()
        self._status.showMessage(f"Opened {path} — {self._db.cell_count()} cells")

    def _on_save(self):
        if self._db._save_path:
            self._db.save_xlsx(self._db._save_path)
            self._status.showMessage(f"Saved to {self._db._save_path}")
        else:
            self._on_save_as()

    def _on_save_as(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Database As", "",
            "Excel Files (*.xlsx);;All Files (*)"
        )
        if not path:
            return
        self._db.save_xlsx(path)
        self._status.showMessage(f"Saved to {path}")

    def _on_export_csv(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        self._db.save_csv(path)
        self._status.showMessage(f"Exported to {path}")

    def _on_import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import CSV", "",
            "CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        self._db.load_csv(path)
        self._refresh_table()
        self._update_combo_options()
        self._status.showMessage(
            f"Imported {path} — {self._db.cell_count()} cells, "
            f"{len(self._db.get_protocol_columns())} protocols"
        )


# ======================================================================
# Entry point
# ======================================================================

def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Database Builder")
    win = DatabaseBuilderWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
