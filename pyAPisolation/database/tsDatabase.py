"""
tsDatabase - cell-level database for intracellular electrophysiology recordings.

Each row is one **cell**.  Columns fall into two categories:

* **Protocol columns** hold file paths (or ``;``-delimited lists of paths)
  linking recordings to the cell.  Within-cell conditions use the naming
  convention ``{protocol} - {condition}`` (e.g. ``IC1 - control``,
  ``IC1 + NE``).
* **Metadata columns** hold cell-level annotations such as ``condition``,
  ``group``, ``drug``, ``experimenter``, ``date``, ``notes``.

The database round-trips through Excel (``.xlsx``) and CSV so that
bench scientists can open and hand-edit the file.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Separator between protocol base name and condition in column headers
CONDITION_SEP = " - "

# Default metadata column names recognised on import
DEFAULT_METADATA_COLS = [
    "condition", "group", "drug", "experimenter", "date", "notes",
    "sex", "age", "animal_id", "cell_type", "well",
]


# ======================================================================
# experimentalStructure - protocol & column-role registry
# ======================================================================

class experimentalStructure:
    """Track which columns are protocols vs metadata and manage conditions."""

    def __init__(self):
        # protocol name -> {altnames: [...], conditions: [...], ...}
        self._protocols: Dict[str, dict] = {}
        # set of column names that are metadata (not protocols)
        self._metadata_cols: set = set()
        self.primary: Optional[str] = None

    # -- protocols ---------------------------------------------------------

    def add_protocol(self, name: str, altnames: Optional[list] = None,
                     conditions: Optional[list] = None, **flags):
        """Register a protocol (idempotent)."""
        entry = self._protocols.setdefault(name, {
            "altnames": [],
            "conditions": [],
        })
        if altnames:
            for a in altnames:
                if a not in entry["altnames"]:
                    entry["altnames"].append(a)
        if conditions:
            for c in conditions:
                if c not in entry["conditions"]:
                    entry["conditions"].append(c)
        entry.update(flags)

    def get_protocol(self, name: str) -> Optional[dict]:
        """Look up by name or altname; return the entry dict or *None*."""
        if name in self._protocols:
            return self._protocols[name]
        for pname, entry in self._protocols.items():
            if name in entry.get("altnames", []):
                return entry
        return None

    def remove_protocol(self, name: str):
        self._protocols.pop(name, None)

    def protocol_names(self) -> List[str]:
        return list(self._protocols.keys())

    def set_primary(self, name: str):
        self.primary = name

    # -- metadata columns --------------------------------------------------

    def mark_metadata(self, col: str):
        self._metadata_cols.add(col)

    def unmark_metadata(self, col: str):
        self._metadata_cols.discard(col)

    def is_metadata(self, col: str) -> bool:
        return col in self._metadata_cols

    def metadata_columns(self) -> List[str]:
        return sorted(self._metadata_cols)

    # -- serialisation helpers ---------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Serialise protocol registry to a DataFrame for saving."""
        rows = []
        for name, entry in self._protocols.items():
            rows.append({
                "name": name,
                "altnames": ";".join(entry.get("altnames", [])),
                "conditions": ";".join(entry.get("conditions", [])),
            })
        if not rows:
            return pd.DataFrame(columns=["name", "altnames", "conditions"])
        return pd.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "experimentalStructure":
        """Reconstruct from a DataFrame (e.g. the *Protocols* sheet)."""
        exp = cls()
        if df is None or df.empty:
            return exp
        for _, row in df.iterrows():
            name = str(row.get("name", ""))
            if not name:
                continue
            raw_alt = row.get("altnames", "")
            altnames = [a for a in str(raw_alt).split(";") if a] if pd.notna(raw_alt) else []
            raw_cond = row.get("conditions", "")
            conditions = [c for c in str(raw_cond).split(";") if c] if pd.notna(raw_cond) else []
            exp.add_protocol(name, altnames=altnames, conditions=conditions)
        return exp

    # -- backward-compat shims ---------------------------------------------

    @property
    def protocols(self) -> pd.DataFrame:
        """Legacy accessor - returns a DataFrame view."""
        return self.to_dataframe()

    @protocols.setter
    def protocols(self, df: pd.DataFrame):
        """Legacy setter - rebuild from DataFrame."""
        self._protocols.clear()
        if df is not None and not df.empty:
            for _, row in df.iterrows():
                name = str(row.get("name", ""))
                if not name:
                    continue
                raw_alt = row.get("altnames", "")
                altnames = [a for a in str(raw_alt).split(";") if a] if pd.notna(raw_alt) else []
                self.add_protocol(name, altnames=altnames)

    def addProtocol(self, name, flags=None, **kw):
        """Legacy wrapper."""
        flags = flags or {}
        altnames = flags.get("altnames", kw.get("altnames"))
        if isinstance(altnames, np.ndarray):
            altnames = altnames.tolist()
        if isinstance(altnames, str):
            altnames = [altnames]
        self.add_protocol(name, altnames=altnames)

    def getProtocol(self, name):
        """Legacy wrapper - returns a one-row DataFrame or None."""
        entry = self.get_protocol(name)
        if entry is None:
            return None
        return pd.DataFrame([{"name": name, **entry}])

    def setPrimary(self, name):
        self.set_primary(name)


# ======================================================================
# tsDatabase
# ======================================================================

class tsDatabase:
    """Cell-level database mapping cells -> protocols -> recording files.

    The main data structure is ``cellindex``, a :class:`pandas.DataFrame`
    where each row is a cell and columns are either *protocol* columns
    (holding file paths) or *metadata* columns (holding annotations).

    Protocol columns may encode within-cell conditions using the naming
    convention ``{base_protocol} - {condition}`` (see ``CONDITION_SEP``).
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, path: Optional[str] = None,
                 exp: Optional[experimentalStructure] = None):
        self.path: str = path or os.getcwd()
        self.exp: experimentalStructure = exp or experimentalStructure()
        self.cellindex: pd.DataFrame = pd.DataFrame()
        self.cellindex.index.name = "cell"
        self._save_path: Optional[str] = None  # last-used save location

    # ------------------------------------------------------------------
    # Cell CRUD
    # ------------------------------------------------------------------

    def add_cell(self, name: str, metadata: Optional[dict] = None):
        """Add a new cell row.  *metadata* sets cell-level columns."""
        if name in self.cellindex.index:
            logger.warning("Cell %r already exists", name)
            return
        row: Dict[str, Any] = {}
        if metadata:
            for k, v in metadata.items():
                row[k] = v
                self.exp.mark_metadata(k)
        new_row = pd.DataFrame([row], index=pd.Index([name], name="cell"))
        # align columns
        for col in self.cellindex.columns:
            if col not in new_row.columns:
                new_row[col] = None
        self.cellindex = pd.concat([self.cellindex, new_row])

    def remove_cell(self, name: str):
        """Remove a cell row."""
        self.cellindex = self.cellindex.drop(index=name, errors="ignore")

    def rename_cell(self, old: str, new: str):
        """Rename a cell (index label)."""
        if old not in self.cellindex.index:
            logger.warning("Cell %r not found", old)
            return
        self.cellindex = self.cellindex.rename(index={old: new})

    def cell_names(self) -> List[str]:
        return list(self.cellindex.index)

    def get_cell(self, name: str) -> dict:
        """Return a single cell as a flat dict."""
        if name not in self.cellindex.index:
            return {}
        return self.cellindex.loc[name].to_dict()

    def cell_count(self) -> int:
        return len(self.cellindex)

    # ------------------------------------------------------------------
    # Metadata CRUD
    # ------------------------------------------------------------------

    def set_cell_metadata(self, cell: str, key: str, value):
        """Set a metadata column value for a cell."""
        self.exp.mark_metadata(key)
        if key not in self.cellindex.columns:
            self.cellindex[key] = None
        if cell in self.cellindex.index:
            self.cellindex.loc[cell, key] = value
        else:
            logger.warning("Cell %r not found", cell)

    def get_metadata_columns(self) -> List[str]:
        """Return metadata column names present in cellindex."""
        return [c for c in self.cellindex.columns if self.exp.is_metadata(c)]

    # ------------------------------------------------------------------
    # Protocol / column CRUD
    # ------------------------------------------------------------------

    @staticmethod
    def _col_name(protocol: str, condition: Optional[str] = None) -> str:
        if condition:
            return f"{protocol}{CONDITION_SEP}{condition}"
        return protocol

    def add_protocol(self, name: str, condition: Optional[str] = None):
        """Add a protocol column (creates column if missing)."""
        col = self._col_name(name, condition)
        if col not in self.cellindex.columns:
            self.cellindex[col] = None
        # register in experimental structure
        conds = [condition] if condition else None
        self.exp.add_protocol(name, conditions=conds)

    def remove_protocol(self, name: str, condition: Optional[str] = None):
        """Drop a protocol column."""
        col = self._col_name(name, condition)
        if col in self.cellindex.columns:
            self.cellindex = self.cellindex.drop(columns=[col])
        if not condition:
            self.exp.remove_protocol(name)

    def get_protocol_columns(self) -> List[str]:
        """Return protocol column names (non-metadata) in cellindex."""
        return [c for c in self.cellindex.columns if not self.exp.is_metadata(c)]

    def protocol_base_name(self, col: str) -> str:
        """Extract the base protocol name (strip condition suffix)."""
        if CONDITION_SEP in col:
            return col.split(CONDITION_SEP, 1)[0]
        return col

    def protocol_condition(self, col: str) -> Optional[str]:
        """Extract the condition suffix, or None."""
        if CONDITION_SEP in col:
            return col.split(CONDITION_SEP, 1)[1]
        return None

    # ------------------------------------------------------------------
    # File assignment
    # ------------------------------------------------------------------

    def assign_file(self, cell: str, protocol: str, filepath: str,
                    condition: Optional[str] = None, *, append: bool = False):
        """Assign a recording file to *cell* under *protocol*.

        If *append* is True and the cell already has a value, append with
        ``;`` (multi-file protocol support).
        """
        col = self._col_name(protocol, condition)
        # ensure column exists
        if col not in self.cellindex.columns:
            self.add_protocol(protocol, condition)
        if cell not in self.cellindex.index:
            logger.warning("Cell %r not found - creating it", cell)
            self.add_cell(cell)

        existing = self.cellindex.loc[cell, col]
        if append and pd.notna(existing) and str(existing).strip():
            paths = str(existing).split(";")
            if filepath not in paths:
                paths.append(filepath)
            self.cellindex.loc[cell, col] = ";".join(paths)
        else:
            self.cellindex.loc[cell, col] = filepath

    def unassign_file(self, cell: str, protocol: str,
                      condition: Optional[str] = None):
        """Clear the file assignment for a cell+protocol."""
        col = self._col_name(protocol, condition)
        if col in self.cellindex.columns and cell in self.cellindex.index:
            self.cellindex.loc[cell, col] = None

    def get_file_list(self, cell: str, protocol: str,
                      condition: Optional[str] = None) -> List[str]:
        """Return the list of file paths for a cell+protocol."""
        col = self._col_name(protocol, condition)
        if col not in self.cellindex.columns or cell not in self.cellindex.index:
            return []
        val = self.cellindex.loc[cell, col]
        if pd.isna(val) or not str(val).strip():
            return []
        return [p.strip() for p in str(val).split(";") if p.strip()]

    # ------------------------------------------------------------------
    # Save / Load - Excel (.xlsx)
    # ------------------------------------------------------------------

    def save_xlsx(self, path: str) -> str:
        """Save to a multi-sheet Excel workbook.

        Sheets: *CellIndex*, *Protocols*, *_config*, *_metadata_cols*.
        """
        if not path.endswith(".xlsx"):
            path += ".xlsx"

        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            # main table
            self.cellindex.to_excel(writer, sheet_name="CellIndex", index=True)

            # protocol registry
            self.exp.to_dataframe().to_excel(writer, sheet_name="Protocols", index=False)

            # config
            pd.DataFrame({
                "key": ["version", "created_by", "database_type", "path"],
                "value": ["2.0", "pyAPisolation", "tsDatabase", self.path],
            }).to_excel(writer, sheet_name="_config", index=False)

            # metadata column list (so we know which cols are metadata on load)
            pd.DataFrame({
                "column": sorted(self.exp._metadata_cols),
            }).to_excel(writer, sheet_name="_metadata_cols", index=False)

        self._save_path = path
        logger.info("Database saved to %s", path)
        return path

    def load_xlsx(self, path: str):
        """Load from a multi-sheet Excel workbook."""
        self.cellindex = pd.read_excel(path, sheet_name="CellIndex", index_col=0)
        self.cellindex.index.name = "cell"

        try:
            proto_df = pd.read_excel(path, sheet_name="Protocols")
            self.exp = experimentalStructure.from_dataframe(proto_df)
        except Exception:
            logger.warning("No Protocols sheet found - using empty registry")
            self.exp = experimentalStructure()

        try:
            meta_df = pd.read_excel(path, sheet_name="_metadata_cols")
            for col in meta_df["column"]:
                self.exp.mark_metadata(str(col))
        except Exception:
            # fall back: guess metadata cols by name
            for col in self.cellindex.columns:
                if col.lower() in {m.lower() for m in DEFAULT_METADATA_COLS}:
                    self.exp.mark_metadata(col)

        try:
            cfg = pd.read_excel(path, sheet_name="_config")
            cfg_dict = dict(zip(cfg["key"], cfg["value"]))
            self.path = cfg_dict.get("path", self.path)
        except Exception:
            pass

        self._save_path = path
        logger.info("Database loaded from %s (%d cells)", path, len(self.cellindex))

    # ------------------------------------------------------------------
    # Save / Load - CSV
    # ------------------------------------------------------------------

    def save_csv(self, path: str) -> str:
        """Save the cellindex as a flat CSV."""
        if not path.endswith(".csv"):
            path += ".csv"
        self.cellindex.to_csv(path, index=True)
        logger.info("Database exported to %s", path)
        return path

    def load_csv(self, path: str,
                 cell_id_col: Optional[str] = None,
                 protocol_cols: Optional[List[str]] = None,
                 metadata_cols: Optional[List[str]] = None):
        """Load from a flat CSV.

        If *cell_id_col* is given it becomes the index; otherwise the
        first column is used.  Columns listed in *protocol_cols* are
        treated as protocol columns; those in *metadata_cols* as metadata.
        Unlisted columns are auto-classified.
        """
        df = pd.read_csv(path)
        if cell_id_col and cell_id_col in df.columns:
            df = df.set_index(cell_id_col)
        elif df.columns[0] == "cell" or df.columns[0] == "Unnamed: 0":
            df = df.set_index(df.columns[0])
        df.index.name = "cell"
        self.cellindex = df

        self.exp = experimentalStructure()
        # classify columns
        for col in df.columns:
            if metadata_cols and col in metadata_cols:
                self.exp.mark_metadata(col)
            elif protocol_cols and col in protocol_cols:
                base = self.protocol_base_name(col)
                cond = self.protocol_condition(col)
                self.exp.add_protocol(base, conditions=[cond] if cond else None)
            else:
                # heuristic: if the column name matches common metadata, treat as metadata
                if col.lower() in {m.lower() for m in DEFAULT_METADATA_COLS}:
                    self.exp.mark_metadata(col)
                else:
                    base = self.protocol_base_name(col)
                    cond = self.protocol_condition(col)
                    self.exp.add_protocol(base, conditions=[cond] if cond else None)

        logger.info("Database loaded from CSV %s (%d cells)", path, len(df))

    # ------------------------------------------------------------------
    # Convenience: new empty database
    # ------------------------------------------------------------------

    def clear(self):
        """Reset to an empty database."""
        self.cellindex = pd.DataFrame()
        self.cellindex.index.name = "cell"
        self.exp = experimentalStructure()
        self._save_path = None

    def next_cell_name(self) -> str:
        """Generate the next auto-incremented cell name."""
        n = self.cell_count() + 1
        while f"Cell_{n:03d}" in self.cellindex.index:
            n += 1
        return f"Cell_{n:03d}"

    # ------------------------------------------------------------------
    # Backward-compat shims (old API -> new API)
    # ------------------------------------------------------------------

    def addEntry(self, name: str, paths=None):
        """Legacy: add a cell, optionally with files."""
        self.add_cell(name)
        if paths is not None:
            if isinstance(paths, (str, os.PathLike)):
                paths = [paths]
            for p in paths:
                try:
                    from ..dataset import cellData
                    cd = cellData(str(p))
                    proto = getattr(cd, "protocol", "unknown")
                    self.assign_file(name, proto, str(p))
                except Exception as exc:
                    logger.warning("Could not parse %s: %s", p, exc)

    def addProtocol(self, cell, protocol, **kwargs):
        """Legacy: add protocol column and optionally assign a file path."""
        path = kwargs.pop("path", None)
        self.add_protocol(protocol)
        if path and cell in self.cellindex.index:
            self.assign_file(cell, protocol, path)

    def updateEntry(self, name, **kwargs):
        """Legacy: update cell-level values."""
        for key, val in kwargs.items():
            if key in self.get_protocol_columns():
                self.cellindex.loc[name, key] = val
            else:
                self.set_cell_metadata(name, key, val)

    def save(self, path):
        """Legacy: save to Excel."""
        return self.save_xlsx(path)

    def load_from_excel(self, path):
        """Legacy: load from Excel."""
        self.load_xlsx(path)

    def getEntries(self):
        return self.cellindex.to_dict(orient="records")

    def getCells(self):
        return self.cellindex.to_dict(orient="index")
