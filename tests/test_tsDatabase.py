"""Unit tests for the tsDatabase backend."""

import importlib.util
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Directly load the tsDatabase module without triggering the main package __init__
# (which pulls in ipfx and other heavy deps that may not be installed)
_mod_path = os.path.join(
    os.path.dirname(__file__), "..", "pyAPisolation", "database", "tsDatabase.py"
)
_spec = importlib.util.spec_from_file_location("tsDatabase", _mod_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CONDITION_SEP = _mod.CONDITION_SEP
DEFAULT_METADATA_COLS = _mod.DEFAULT_METADATA_COLS
experimentalStructure = _mod.experimentalStructure
tsDatabase = _mod.tsDatabase


# ======================================================================
# experimentalStructure
# ======================================================================

class TestExperimentalStructure:

    def test_add_and_get_protocol(self):
        exp = experimentalStructure()
        exp.add_protocol("IC1", altnames=["ic1", "IC_long_square"])
        assert "IC1" in exp.protocol_names()
        entry = exp.get_protocol("IC1")
        assert "ic1" in entry["altnames"]

    def test_get_by_altname(self):
        exp = experimentalStructure()
        exp.add_protocol("IC1", altnames=["ic1"])
        assert exp.get_protocol("ic1") is not None

    def test_conditions(self):
        exp = experimentalStructure()
        exp.add_protocol("IC1", conditions=["control", "NE"])
        entry = exp.get_protocol("IC1")
        assert "control" in entry["conditions"]
        assert "NE" in entry["conditions"]

    def test_metadata_tracking(self):
        exp = experimentalStructure()
        exp.mark_metadata("drug")
        assert exp.is_metadata("drug")
        exp.unmark_metadata("drug")
        assert not exp.is_metadata("drug")

    def test_round_trip_dataframe(self):
        exp = experimentalStructure()
        exp.add_protocol("IC1", altnames=["ic1"], conditions=["ctrl"])
        exp.add_protocol("Sag")
        df = exp.to_dataframe()
        exp2 = experimentalStructure.from_dataframe(df)
        assert set(exp2.protocol_names()) == {"IC1", "Sag"}
        assert "ic1" in exp2.get_protocol("IC1")["altnames"]
        assert "ctrl" in exp2.get_protocol("IC1")["conditions"]

    def test_remove_protocol(self):
        exp = experimentalStructure()
        exp.add_protocol("IC1")
        exp.remove_protocol("IC1")
        assert "IC1" not in exp.protocol_names()


# ======================================================================
# tsDatabase — cell CRUD
# ======================================================================

class TestCellCRUD:

    def test_add_and_list(self):
        db = tsDatabase()
        db.add_cell("Cell_001")
        db.add_cell("Cell_002")
        assert db.cell_count() == 2
        assert db.cell_names() == ["Cell_001", "Cell_002"]

    def test_add_with_metadata(self):
        db = tsDatabase()
        db.add_cell("Cell_001", metadata={"drug": "NE", "group": "stress"})
        assert db.get_cell("Cell_001")["drug"] == "NE"
        assert "drug" in db.get_metadata_columns()

    def test_remove_cell(self):
        db = tsDatabase()
        db.add_cell("Cell_001")
        db.remove_cell("Cell_001")
        assert db.cell_count() == 0

    def test_rename_cell(self):
        db = tsDatabase()
        db.add_cell("Cell_001")
        db.rename_cell("Cell_001", "My_Cell")
        assert "My_Cell" in db.cell_names()
        assert "Cell_001" not in db.cell_names()

    def test_duplicate_cell_names(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_cell("C1")  # should warn but not crash
        assert db.cell_count() == 1

    def test_next_cell_name(self):
        db = tsDatabase()
        assert db.next_cell_name() == "Cell_001"
        db.add_cell("Cell_001")
        assert db.next_cell_name() == "Cell_002"


# ======================================================================
# tsDatabase — protocol CRUD
# ======================================================================

class TestProtocolCRUD:

    def test_add_protocol(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        assert "IC1" in db.get_protocol_columns()

    def test_add_protocol_with_condition(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1", condition="control")
        db.add_protocol("IC1", condition="NE")
        cols = db.get_protocol_columns()
        assert "IC1 - control" in cols
        assert "IC1 - NE" in cols

    def test_remove_protocol(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.remove_protocol("IC1")
        assert "IC1" not in db.get_protocol_columns()

    def test_protocol_base_and_condition(self):
        db = tsDatabase()
        assert db.protocol_base_name("IC1 - control") == "IC1"
        assert db.protocol_condition("IC1 - control") == "control"
        assert db.protocol_base_name("IC1") == "IC1"
        assert db.protocol_condition("IC1") is None


# ======================================================================
# tsDatabase — file assignment
# ======================================================================

class TestFileAssignment:

    def test_assign_and_get(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.assign_file("C1", "IC1", "/path/file.abf")
        files = db.get_file_list("C1", "IC1")
        assert files == ["/path/file.abf"]

    def test_assign_with_condition(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.assign_file("C1", "IC1", "/ctrl.abf", condition="control")
        db.assign_file("C1", "IC1", "/ne.abf", condition="NE")
        assert db.get_file_list("C1", "IC1", condition="control") == ["/ctrl.abf"]
        assert db.get_file_list("C1", "IC1", condition="NE") == ["/ne.abf"]

    def test_append_multi_file(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.assign_file("C1", "IC1", "/a.abf")
        db.assign_file("C1", "IC1", "/b.abf", append=True)
        files = db.get_file_list("C1", "IC1")
        assert files == ["/a.abf", "/b.abf"]

    def test_unassign(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.assign_file("C1", "IC1", "/a.abf")
        db.unassign_file("C1", "IC1")
        assert db.get_file_list("C1", "IC1") == []

    def test_assign_creates_cell_if_missing(self):
        db = tsDatabase()
        db.add_protocol("IC1")
        db.assign_file("NewCell", "IC1", "/a.abf")
        assert "NewCell" in db.cell_names()

    def test_empty_get(self):
        db = tsDatabase()
        assert db.get_file_list("missing", "missing") == []


# ======================================================================
# tsDatabase — metadata
# ======================================================================

class TestMetadata:

    def test_set_and_get(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.set_cell_metadata("C1", "drug", "NE")
        assert db.get_cell("C1")["drug"] == "NE"
        assert "drug" in db.get_metadata_columns()

    def test_metadata_not_in_protocol_cols(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.set_cell_metadata("C1", "drug", "NE")
        assert "drug" not in db.get_protocol_columns()
        assert "IC1" not in db.get_metadata_columns()


# ======================================================================
# tsDatabase — XLSX round-trip
# ======================================================================

class TestXLSXRoundTrip:

    def test_save_and_load(self, tmp_path):
        db = tsDatabase()
        db.add_cell("C1", metadata={"drug": "NE", "group": "stress"})
        db.add_cell("C2", metadata={"drug": "aCSF", "group": "control"})
        db.add_protocol("IC1")
        db.add_protocol("Sag")
        db.assign_file("C1", "IC1", "/c1_ic1.abf")
        db.assign_file("C2", "Sag", "/c2_sag.abf")
        db.set_cell_metadata("C1", "notes", "good cell")

        path = str(tmp_path / "test.xlsx")
        db.save_xlsx(path)
        assert os.path.isfile(path)

        # Reload
        db2 = tsDatabase()
        db2.load_xlsx(path)
        assert db2.cell_count() == 2
        assert set(db2.cell_names()) == {"C1", "C2"}
        assert db2.get_file_list("C1", "IC1") == ["/c1_ic1.abf"]
        assert db2.get_cell("C1")["notes"] == "good cell"
        assert "drug" in db2.get_metadata_columns()
        assert "IC1" in db2.get_protocol_columns()

    def test_condition_columns_survive_roundtrip(self, tmp_path):
        db = tsDatabase()
        db.add_cell("C1")
        db.assign_file("C1", "IC1", "/ctrl.abf", condition="control")
        db.assign_file("C1", "IC1", "/ne.abf", condition="NE")

        path = str(tmp_path / "cond.xlsx")
        db.save_xlsx(path)

        db2 = tsDatabase()
        db2.load_xlsx(path)
        assert "IC1 - control" in db2.get_protocol_columns()
        assert "IC1 - NE" in db2.get_protocol_columns()
        assert db2.get_file_list("C1", "IC1", condition="control") == ["/ctrl.abf"]


# ======================================================================
# tsDatabase — CSV round-trip
# ======================================================================

class TestCSVRoundTrip:

    def test_save_and_load(self, tmp_path):
        db = tsDatabase()
        db.add_cell("C1", metadata={"group": "stress"})
        db.add_protocol("IC1")
        db.assign_file("C1", "IC1", "/file.abf")

        path = str(tmp_path / "test.csv")
        db.save_csv(path)
        assert os.path.isfile(path)

        db2 = tsDatabase()
        db2.load_csv(path)
        assert db2.cell_count() == 1
        assert "C1" in db2.cell_names()


# ======================================================================
# tsDatabase — clear
# ======================================================================

class TestClear:

    def test_clear(self):
        db = tsDatabase()
        db.add_cell("C1")
        db.add_protocol("IC1")
        db.clear()
        assert db.cell_count() == 0
        assert db.get_protocol_columns() == []
