"""
pyAPisolation.database — cell-level database for intracellular ephys recordings.

Core classes
------------
- ``tsDatabase``: DataFrame-backed database mapping cells → protocols → files
- ``experimentalStructure``: Protocol registry tracking column roles and conditions
"""

from .tsDatabase import tsDatabase, experimentalStructure, CONDITION_SEP, DEFAULT_METADATA_COLS

__all__ = ["tsDatabase", "experimentalStructure", "CONDITION_SEP", "DEFAULT_METADATA_COLS"]
