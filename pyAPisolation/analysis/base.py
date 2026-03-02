"""
Base class for all analysis modules.

Subclass ``AnalysisBase`` and implement **one method** — ``analyze()`` — to
create a new analysis.  The framework handles input normalization, sweep
iteration, batching, and result packaging automatically.

Quickstart
----------
::

    from pyAPisolation.analysis import AnalysisBase, register
    import numpy as np

    class PeakDetector(AnalysisBase):
        \"\"\"Find the peak voltage in each sweep.\"\"\"
        name = "peak_detector"
        sweep_mode = "per_sweep"          # framework calls analyze() once per sweep

        # Parameters — just typed class attributes
        min_voltage: float = -20.0

        def analyze(self, x, y, c, **kwargs):
            peak_v = float(np.max(y))
            peak_t = float(x[np.argmax(y)])
            if peak_v < self.min_voltage:
                return {"peak_found": False}
            return {"peak_voltage": peak_v, "peak_time": peak_t, "peak_found": True}

    register(PeakDetector)
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .result import AnalysisResult

logger = logging.getLogger(__name__)

# Sentinel used to distinguish "user params" from "internal attrs"
_INTERNAL_ATTRS = frozenset({
    "name", "display_name", "sweep_mode",
    # private / dunder attrs are filtered automatically
})


class AnalysisBase:
    """
    Base class for analysis modules.

    Subclasses **must** implement :meth:`analyze`.

    Class attributes
    ----------------
    name : str
        Short identifier used for registry look-up (e.g. ``"spike"``).
        Defaults to the lower-cased class name.
    display_name : str
        Human-readable label (e.g. ``"Spike Analysis"``).
        Defaults to *name*.
    sweep_mode : str
        ``"per_sweep"`` — framework iterates sweeps and calls *analyze()*
        with 1-D arrays for each sweep, then aggregates.
        ``"per_file"`` — *analyze()* receives 2-D arrays
        ``(sweeps × samples)`` and is responsible for its own iteration.

    Parameters
    ----------
    Define parameters as **typed class attributes** with defaults::

        dv_cutoff: float = 7.0
        min_peak: float = -10.0

    The framework discovers them via ``__annotations__`` and provides
    :meth:`get_parameters`, :meth:`set_parameters`,
    and :meth:`_collect_param_dict` automatically.
    """

    # ---- class-level configuration ----
    name: str = ""
    display_name: str = ""
    sweep_mode: str = "per_sweep"  # "per_sweep" or "per_file"

    def __init__(self, **overrides):
        """
        Instantiate the module, optionally overriding parameter defaults.

        Parameters
        ----------
        **overrides
            Parameter values to override, e.g. ``SpikeAnalysis(dv_cutoff=10)``.
        """
        # Derive name / display_name if not set
        if not self.name:
            self.name = type(self).__name__.lower()
        if not self.display_name:
            self.display_name = self.name

        # Apply any overrides the user passed in
        for key, value in overrides.items():
            if key in self._param_names():
                setattr(self, key, value)
            else:
                logger.warning(
                    f"{self.name}: unknown parameter '{key}' ignored"
                )

    # ==================================================================
    # Abstract method — the only thing users MUST implement
    # ==================================================================

    def analyze(self, x, y, c, **kwargs) -> dict:
        """
        Run the analysis on one unit of data.

        Parameters
        ----------
        x : np.ndarray
            Time array.  1-D in *per_sweep* mode, 2-D in *per_file* mode.
        y : np.ndarray
            Voltage / response array (same shape as *x*).
        c : np.ndarray
            Current / command array (same shape as *x*).
        **kwargs
            Extra context injected by the framework:

            * ``sweep_number`` (int) — current sweep index (per_sweep only)
            * ``file_path`` (str)   — source file path, if available
            * ``celldata``          — the ``cellData`` object, if available

        Returns
        -------
        dict
            A flat dictionary of results.
            Keys become column names in the output DataFrame.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement analyze()"
        )

    # ==================================================================
    # Framework entry point — handles input, sweep loop, result wrapping
    # ==================================================================

    def run(
        self,
        x=None, y=None, c=None,
        file=None,
        celldata=None,
        selected_sweeps=None,
        **kwargs
    ) -> AnalysisResult:
        """
        Main entry point. Accepts raw arrays, a file path, or a cellData
        object.  The framework normalizes the input, iterates sweeps
        (if ``sweep_mode == "per_sweep"``), and wraps the output in an
        :class:`AnalysisResult`.

        Parameters
        ----------
        x, y, c : np.ndarray, optional
            Raw data arrays (1-D or 2-D).
        file : str, optional
            Path to an ABF file.
        celldata : cellData, optional
            Pre-loaded data container.
        selected_sweeps : list[int], optional
            Subset of sweeps to analyze.  ``None`` ⇒ all sweeps.

        Returns
        -------
        AnalysisResult
        """
        from ..patch_utils import parse_user_input  # lazy to avoid circular

        # --- resolve input to a cellData ----------------------------------
        if celldata is not None:
            data = celldata
        else:
            data = parse_user_input(x, y, c, file)

        file_path = getattr(data, "filePath", None) or \
                    getattr(data, "file", None) or \
                    (file if isinstance(file, str) else "array_input")

        result = AnalysisResult(
            name=self.name,
            file_path=file_path,
            success=True,
        )

        # --- determine sweeps to process ----------------------------------
        if selected_sweeps is None:
            selected_sweeps = list(range(data.sweepCount))

        result.metadata["sweep_count"] = len(selected_sweeps)
        result.metadata["protocol"] = getattr(data, "protocol", "")

        # --- run analysis -------------------------------------------------
        try:
            if self.sweep_mode == "per_sweep":
                result = self._run_per_sweep(data, selected_sweeps, result, **kwargs)
            elif self.sweep_mode == "per_file":
                result = self._run_per_file(data, selected_sweeps, result, **kwargs)
            else:
                raise ValueError(
                    f"Unknown sweep_mode '{self.sweep_mode}'. "
                    f"Use 'per_sweep' or 'per_file'."
                )
        except Exception as exc:
            result.add_error(f"{type(exc).__name__}: {exc}")
            logger.exception("Analysis failed for %s", file_path)

        return result

    # ------------------------------------------------------------------
    # Internal sweep-dispatch helpers
    # ------------------------------------------------------------------

    def _run_per_sweep(self, data, sweeps, result, **kwargs) -> AnalysisResult:
        """Call analyze() once per sweep with 1-D arrays."""
        for sweep_idx in sweeps:
            data.setSweep(sweep_idx)
            sweep_x = np.asarray(data.sweepX, dtype=np.float64)
            sweep_y = np.asarray(data.sweepY, dtype=np.float64)
            sweep_c = np.asarray(data.sweepC, dtype=np.float64)

            try:
                out = self.analyze(
                    sweep_x, sweep_y, sweep_c,
                    sweep_number=sweep_idx,
                    file_path=result.file_path,
                    celldata=data,
                    **kwargs,
                )
            except Exception as exc:
                result.add_warning(
                    f"Sweep {sweep_idx} failed: {type(exc).__name__}: {exc}"
                )
                out = {"_error": str(exc)}

            if not isinstance(out, dict):
                # If the user returned a DataFrame, convert to dict
                if isinstance(out, pd.DataFrame):
                    out = out.to_dict(orient="list")
                else:
                    out = {"result": out}

            out["sweep_number"] = sweep_idx
            result.sweep_results.append(out)

        return result

    def _run_per_file(self, data, sweeps, result, **kwargs) -> AnalysisResult:
        """Call analyze() once with full 2-D arrays."""
        # Build 2-D arrays for the selected sweeps
        xs, ys, cs = [], [], []
        for sweep_idx in sweeps:
            data.setSweep(sweep_idx)
            xs.append(np.asarray(data.sweepX, dtype=np.float64))
            ys.append(np.asarray(data.sweepY, dtype=np.float64))
            cs.append(np.asarray(data.sweepC, dtype=np.float64))

        x_2d = np.array(xs)
        y_2d = np.array(ys)
        c_2d = np.array(cs)

        try:
            out = self.analyze(
                x_2d, y_2d, c_2d,
                file_path=result.file_path,
                celldata=data,
                selected_sweeps=sweeps,
                **kwargs,
            )
        except Exception as exc:
            result.add_error(f"{type(exc).__name__}: {exc}")
            return result

        if isinstance(out, dict):
            result.data = out
        elif isinstance(out, pd.DataFrame):
            result.data = out.to_dict(orient="list")
        else:
            result.data = {"result": out}

        return result

    # ==================================================================
    # Parameter introspection
    # ==================================================================

    @classmethod
    def _param_names(cls) -> List[str]:
        """
        Discover user-defined parameter names by inspecting annotations
        on this class and all its bases (excluding AnalysisBase internals).
        """
        params = []
        # Walk MRO (most specific first) and collect annotated attrs
        for klass in cls.__mro__:
            if klass is object:
                continue
            for attr_name in getattr(klass, "__annotations__", {}):
                if attr_name.startswith("_"):
                    continue
                if attr_name in _INTERNAL_ATTRS:
                    continue
                if attr_name not in params:
                    params.append(attr_name)
        return params

    def get_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a dict describing all user-defined parameters.

        Returns
        -------
        dict
            ``{param_name: {"type": <type>, "default": <val>, "value": <val>}}``
        """
        info = {}
        for name in self._param_names():
            ann = self._get_annotation(name)
            default = self._get_class_default(name)
            info[name] = {
                "type": ann,
                "default": default,
                "value": getattr(self, name, default),
            }
        return info

    def set_parameters(self, **kwargs) -> None:
        """
        Update parameter values with basic type coercion.

        Parameters
        ----------
        **kwargs
            ``name=value`` pairs.
        """
        for key, value in kwargs.items():
            if key not in self._param_names():
                logger.warning(f"{self.name}: unknown parameter '{key}'")
                continue
            expected = self._get_annotation(key)
            if expected is not None and not isinstance(value, expected):
                try:
                    value = expected(value)
                except (ValueError, TypeError):
                    logger.warning(
                        f"{self.name}: cannot coerce '{key}' value "
                        f"{value!r} to {expected.__name__}"
                    )
                    continue
            setattr(self, key, value)

    def _collect_param_dict(self) -> dict:
        """
        Build a plain dict of current parameter values.

        Useful for passing to legacy functions that expect ``param_dict``.
        """
        return {
            name: getattr(self, name)
            for name in self._param_names()
        }

    # ------------------------------------------------------------------
    # Annotation helpers
    # ------------------------------------------------------------------

    def _get_annotation(self, name: str):
        """Get the type annotation for a parameter, or None."""
        for klass in type(self).__mro__:
            anns = getattr(klass, "__annotations__", {})
            if name in anns:
                return anns[name]
        return None

    @classmethod
    def _get_class_default(cls, name: str):
        """Get the class-level default for a parameter."""
        for klass in cls.__mro__:
            if name in klass.__dict__:
                return klass.__dict__[name]
        return None

    # ==================================================================
    # Dunder helpers
    # ==================================================================

    def __str__(self):
        return f"{type(self).__name__}(name='{self.name}', mode='{self.sweep_mode}')"

    def __repr__(self):
        params = self._collect_param_dict()
        return (f"{type(self).__name__}(name='{self.name}', "
                f"mode='{self.sweep_mode}', params={params})")
