"""
Lightweight result container for analysis outputs.

AnalysisResult wraps whatever dict your analyze() method returns
and provides helpers for aggregation, DataFrame export, and serialization.
"""

from collections import OrderedDict
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

    # Extra named output sheets (e.g. a module-specific summary).  These are
    # merged ahead of the generic "Raw" / "Summary" sheets by :meth:`to_sheets`.
    sheets: Dict[str, pd.DataFrame] = field(default_factory=dict)

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

    # ------------------------------------------------------------------
    # DataFrame export
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return the "raw" DataFrame -- one row per sweep (or per file).

        For a combined/batch result this returns the pre-built DataFrame
        stashed by :meth:`concatenate`; otherwise it is built on the fly:

        * If there are *sweep_results*, each sweep becomes a row and the
          file-level ``data`` dict is broadcast across all rows.
        * Otherwise a single-row DataFrame is built from ``data``.
        """
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
    # Summary / multi-sheet export
    # ------------------------------------------------------------------

    def summary_dataframe(self) -> pd.DataFrame:
        """
        Build a per-file summary: one row per source file.

        The raw DataFrame (one row per sweep) is grouped by its ``file``
        column and every *numeric scalar* column is reduced to its mean.
        List-valued columns (e.g. per-spike arrays like ``peak_t``) are
        dropped automatically because they are not numeric dtypes.

        Two extra columns are added:

        * ``n_sweeps`` -- number of raw rows that went into each file.
        * ``total_spike_count`` -- sum of ``spike_count`` per file, when a
          ``spike_count`` column is present.

        Returns an empty DataFrame if there is no data or no ``file`` column.
        """
        raw = self.to_dataframe()
        if raw is None or raw.empty or "file" not in raw.columns:
            return pd.DataFrame()

        grouped = raw.groupby("file", sort=False)

        numeric = raw.select_dtypes(include="number")
        # "sweep" is a row index, not a feature -- keep it out of the means.
        numeric = numeric.drop(columns=[c for c in ("sweep",) if c in numeric.columns])

        if numeric.shape[1] > 0:
            summary = numeric.groupby(raw["file"], sort=False).mean()
        else:
            # No numeric features -- still emit one row per file.
            summary = pd.DataFrame(index=grouped.size().index)

        summary.insert(0, "n_sweeps", grouped.size())

        if "spike_count" in raw.columns:
            summary["total_spike_count"] = grouped["spike_count"].sum()

        return summary.reset_index()

    def to_sheets(self) -> "OrderedDict[str, pd.DataFrame]":
        """
        Return the named sheets to display / export, in tab order.

        Always includes a generic ``"Summary"`` (one row per file) followed
        by ``"Raw"`` (one row per sweep).  Any module-supplied entries in
        :attr:`sheets` are inserted ahead of these so a module can override
        or augment the defaults.
        """
        out: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
        for key, frame in self.sheets.items():
            out[key] = frame
        out.setdefault("Summary", self.summary_dataframe())
        out.setdefault("Raw", self.to_dataframe())
        return out

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
