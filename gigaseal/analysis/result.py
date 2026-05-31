"""
Lightweight result container for analysis outputs.

AnalysisResult wraps whatever dict your analyze() method returns
and provides helpers for aggregation, DataFrame export, and serialization.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """
    Container for the output of a single analysis run.

    Attributes:
        name:          Name of the analysis module that produced this result.
        file_path:     Path to the source file (or 'array_input' for raw data).
        success:       Whether the analysis completed without error.
        data:          The dict returned by analyze() (per-file mode) or an
                       aggregated dict built from sweep_results.
        sweep_results: List of per-sweep dicts (populated in per_sweep mode).
        errors:        Any error messages captured during the run.
        warnings:      Any warning messages captured during the run.
        metadata:      Extra info (sweep count, protocol, etc.).
    """

    name: str
    file_path: str = "unknown"
    success: bool = True

    # Main data -- the dict your analyze() returned
    data: Dict[str, Any] = field(default_factory=dict)

    # Per-sweep results (list of dicts, one per sweep)
    sweep_results: List[Dict[str, Any]] = field(default_factory=list)

    # Diagnostics
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def add_error(self, msg: str) -> None:
        """Record an error and mark the result as failed."""
        self.errors.append(msg)
        self.success = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    # ------------------------------------------------------------------
    # DataFrame export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Flatten result data into a single DataFrame.

        * If there are *sweep_results*, each sweep becomes a row and the
          file-level ``data`` dict is broadcast across all rows.
        * Otherwise a single-row DataFrame is built from ``data``.
        """
        if self.sweep_results:
            rows = []
            for i, sweep_dict in enumerate(self.sweep_results):
                row = {"file": self.file_path, "sweep": i}
                row.update(self.data)          # file-level columns
                row.update(sweep_dict)         # sweep-level columns
                rows.append(row)
            return pd.DataFrame(rows)
        else:
            row = {"file": self.file_path}
            row.update(self.data)
            return pd.DataFrame([row])

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @classmethod
    def concatenate(cls, results: List["AnalysisResult"]) -> "AnalysisResult":
        """
        Merge a list of AnalysisResult objects into one combined result.

        The combined ``data`` dict is empty; all information lives in the
        concatenated DataFrame accessible via ``to_dataframe()``.
        """
        if not results:
            return cls(name="empty", file_path="none", success=False,
                       errors=["No results to concatenate"])

        # Build a combined DataFrame from all individual results
        frames = [r.to_dataframe() for r in results if r.success]
        combined_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        all_errors = [e for r in results for e in r.errors]
        all_warnings = [w for r in results for w in r.warnings]
        all_files = [r.file_path for r in results]

        combined = cls(
            name=results[0].name,
            file_path=str(all_files),
            success=all(r.success for r in results),
            data={},
            sweep_results=[],
            errors=all_errors,
            warnings=all_warnings,
            metadata={"file_count": len(results), "files": all_files},
        )
        # Stash the pre-built DataFrame so to_dataframe() returns it
        combined._combined_df = combined_df
        return combined

    def to_dataframe(self) -> pd.DataFrame:  # noqa: F811  -- intentional redefinition
        """Return the DataFrame, using the pre-built one if available."""
        if hasattr(self, "_combined_df"):
            return self._combined_df
        return self._build_dataframe()

    def _build_dataframe(self) -> pd.DataFrame:
        """Build a DataFrame from this single result."""
        if self.sweep_results:
            rows = []
            for i, sweep_dict in enumerate(self.sweep_results):
                row = {"file": self.file_path, "sweep": i}
                row.update(self.data)
                row.update(sweep_dict)
                rows.append(row)
            return pd.DataFrame(rows)
        else:
            row = {"file": self.file_path}
            row.update(self.data)
            return pd.DataFrame([row])

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "file_path": self.file_path,
            "success": self.success,
            "data": self.data,
            "sweep_results": self.sweep_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        status = "OK" if self.success else f"FAILED ({len(self.errors)} errors)"
        n_sweeps = len(self.sweep_results)
        return (f"AnalysisResult(name='{self.name}', file='{self.file_path}', "
                f"{status}, sweeps={n_sweeps})")
