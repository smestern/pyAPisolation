"""
AnalysisController — glue between the GUI panels and the
``pyAPisolation.analysis`` framework.

Handles file loading, dispatching individual / batch analysis in a worker
thread, progress reporting, and legacy fallback for the built-in spike
and subthreshold modules.
"""

from __future__ import annotations

import copy
import os
import time
import traceback
from typing import Optional

import numpy as np
import pandas as pd
from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot, Qt
from PySide6.QtWidgets import QMessageBox, QProgressDialog


# ---------------------------------------------------------------------------
# Worker for running analysis off the main thread
# ---------------------------------------------------------------------------

class _WorkerSignals(QObject):
    finished = Signal(object)  # result payload
    error = Signal(str)
    progress = Signal(int, int)  # current, total


class _AnalysisWorker(QRunnable):
    """Run a callable in the thread pool, reporting progress."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = _WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.error.emit(f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


class AnalysisController(QObject):
    """
    Coordinates analysis execution for the new GUI.

    Connects to panel signals and dispatches work to the analysis
    framework (modular-first, legacy-fallback).
    """

    # Forwarded to the UI after analysis completes
    individual_analysis_done = Signal(object)  # AnalysisResult or legacy dict
    batch_analysis_done = Signal(object)       # AnalysisResult or DataFrame
    analysis_error = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._celldata = None  # currently loaded cellData
        self._current_file: str = ""

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def load_file(self, file_path: str):
        """Load an ABF file into a cellData container."""
        from pyAPisolation.dataset import cellData
        self._celldata = cellData(file_path)
        self._current_file = file_path
        return self._celldata

    @property
    def celldata(self):
        return self._celldata

    # ------------------------------------------------------------------
    # Individual analysis (single file, selected sweeps)
    # ------------------------------------------------------------------

    def run_individual(
        self,
        module_name: str,
        params: dict,
        sweeps: list[int],
        show_rejected: bool = False,
    ):
        """
        Run analysis on the currently loaded file.

        Tries the modular ``AnalysisBase.run()`` first; falls back to
        legacy feature-extractor functions for ``spike`` and
        ``subthreshold``.

        Returns the result synchronously (called from main thread but
        fast enough for single-file analysis).
        """
        if self._celldata is None:
            self.analysis_error.emit("No file loaded.")
            return None

        # --- Try modular path -----------------------------------------
        module = self._get_module(module_name)
        if module is not None:
            try:
                module.set_parameters(**params)
                result = module.run(
                    celldata=self._celldata,
                    selected_sweeps=sweeps,
                )
                # Attach rejected spikes if requested
                if show_rejected and module_name in ("spike",):
                    rejected = self._compute_rejected(params, sweeps)
                    result.metadata["rejected"] = rejected
                self.individual_analysis_done.emit(result)
                return result
            except Exception as exc:
                print(f"[Controller] Modular analysis failed, falling back to legacy: {exc}")

        # --- Legacy fallback ------------------------------------------
        return self._run_individual_legacy(module_name, params, sweeps, show_rejected)

    def _run_individual_legacy(self, module_name, params, sweeps, show_rejected):
        """Legacy per-sweep spike/subthreshold analysis."""
        from pyAPisolation.featureExtractor import analyze_subthres

        result_payload = {
            "spike_df": None,
            "rejected_spikes": None,
            "subthres_df": None,
        }

        if module_name in ("spike", "peak_detector"):
            result_payload.update(
                self._legacy_spike_individual(params, sweeps, show_rejected)
            )
        elif module_name in ("subthreshold", "subthres"):
            try:
                start = params.get("start", 0.0)
                end = params.get("end", 0.0)
                df, _ = analyze_subthres(
                    file=self._current_file, start=start, end=end
                )
                result_payload["subthres_df"] = df
            except Exception as exc:
                self.analysis_error.emit(f"Subthreshold analysis failed: {exc}")

        self.individual_analysis_done.emit(result_payload)
        return result_payload

    def _legacy_spike_individual(self, params, sweeps, show_rejected):
        """Run ipfx SpikeFeatureExtractor per sweep (mirrors spikeFinder.py)."""
        from ipfx.feature_extractor import SpikeFeatureExtractor
        from pyAPisolation.featureExtractor import determine_rejected_spikes

        p = copy.deepcopy(params)
        cd = self._celldata

        if p.get("end", 0) == 0 or p.get("end", 0) > cd.sweepX[-1]:
            p["end"] = float(cd.sweepX[-1])

        spfx = SpikeFeatureExtractor(
            filter=0,
            dv_cutoff=p.get("dv_cutoff", 7.0),
            max_interval=p.get("max_interval", 0.005),
            min_height=p.get("min_height", 2.0),
            min_peak=p.get("min_peak", -10.0),
            start=p.get("start", 0.0),
            end=p["end"],
            thresh_frac=p.get("thresh_frac", 0.2),
        )

        spike_df = {}
        rejected_spikes = {} if show_rejected else None
        for sweep in sweeps:
            cd.setSweep(sweep)
            spike_df[sweep] = spfx.process(cd.sweepX, cd.sweepY, cd.sweepC)
            if show_rejected:
                rej = determine_rejected_spikes(
                    spfx, spike_df[sweep], cd.sweepY, cd.sweepX, p
                )
                rejected_spikes[sweep] = pd.DataFrame.from_dict(rej).T

        return {"spike_df": spike_df, "rejected_spikes": rejected_spikes}

    def _compute_rejected(self, params, sweeps):
        """Compute rejected spikes using the legacy helper."""
        try:
            from ipfx.feature_extractor import SpikeFeatureExtractor
            from pyAPisolation.featureExtractor import determine_rejected_spikes

            p = copy.deepcopy(params)
            cd = self._celldata
            if p.get("end", 0) == 0:
                p["end"] = float(cd.sweepX[-1])

            spfx = SpikeFeatureExtractor(
                filter=0,
                dv_cutoff=p.get("dv_cutoff", 7.0),
                max_interval=p.get("max_interval", 0.005),
                min_height=p.get("min_height", 2.0),
                min_peak=p.get("min_peak", -10.0),
                start=p.get("start", 0.0),
                end=p["end"],
                thresh_frac=p.get("thresh_frac", 0.2),
            )

            rejected = {}
            for sweep in sweeps:
                cd.setSweep(sweep)
                spike_df = spfx.process(cd.sweepX, cd.sweepY, cd.sweepC)
                rej = determine_rejected_spikes(
                    spfx, spike_df, cd.sweepY, cd.sweepX, p
                )
                rejected[sweep] = pd.DataFrame.from_dict(rej).T
            return rejected
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Batch analysis (whole folder)
    # ------------------------------------------------------------------

    def run_batch(
        self,
        folder: str,
        module_name: str,
        params: dict,
        protocol_filter: str = "",
        n_jobs: int = 1,
        output_tag: str = "",
        parent_widget=None,
    ):
        """
        Run batch analysis in a worker thread.

        Shows a blocking QProgressDialog and emits ``batch_analysis_done``
        with the result.
        """
        module = self._get_module(module_name)
        if module is None:
            self.analysis_error.emit(f"Module '{module_name}' not found in registry.")
            return

        module.set_parameters(**params)

        # Progress dialog
        progress = QProgressDialog("Running batch analysis…", "Cancel", 0, 0, parent_widget)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()

        def _progress_cb(done, total):
            progress.setMaximum(total)
            progress.setValue(done)

        def _run():
            from pyAPisolation.analysis.runner import run_batch, save_results

            proto = protocol_filter if protocol_filter and protocol_filter != "[No Filter]" else None
            result = run_batch(
                module, folder,
                protocol_filter=proto,
                selected_sweeps=None,
                n_jobs=n_jobs,
                progress_callback=_progress_cb,
            )

            # Save to disk
            tag = output_tag or str(int(time.time()))
            try:
                save_results(result, folder, tag=tag, fmt="csv")
            except Exception:
                pass

            return result

        worker = _AnalysisWorker(_run)

        def _on_done(result):
            progress.close()
            self.batch_analysis_done.emit(result)

        def _on_error(msg):
            progress.close()
            # Try legacy fallback
            legacy_result = self._run_batch_legacy(
                folder, module_name, params, protocol_filter, output_tag
            )
            if legacy_result is not None:
                self.batch_analysis_done.emit(legacy_result)
            else:
                self.analysis_error.emit(msg)

        worker.signals.finished.connect(_on_done)
        worker.signals.error.connect(_on_error)
        self._pool.start(worker)

    def _run_batch_legacy(self, folder, module_name, params, protocol_filter, output_tag):
        """Legacy batch fallback using featureExtractor functions."""
        try:
            if module_name in ("spike", "peak_detector"):
                from pyAPisolation.featureExtractor import process_file, save_data_frames
                import glob as _glob

                filelist = sorted(_glob.glob(os.path.join(folder, "**/*.abf"), recursive=True))
                spike_counts, full_dfs, running_avgs = [], [], []

                for f in filelist:
                    try:
                        sc, fd, ra = process_file(f, params, protocol_filter or None)
                        spike_counts.append(sc)
                        full_dfs.append(fd)
                        running_avgs.append(ra)
                    except Exception:
                        continue

                if spike_counts:
                    df_spike = pd.concat(spike_counts, sort=True)
                    df_full = pd.concat(full_dfs, sort=True)
                    df_run = pd.concat(running_avgs, sort=False)
                    tag = output_tag or str(int(time.time()))
                    save_data_frames(df_spike, df_full, df_run, folder, tag,
                                     True, True, True)
                    return df_full
            elif module_name in ("subthreshold", "subthres"):
                from pyAPisolation.featureExtractor import (
                    preprocess_abf_subthreshold, save_subthres_data,
                )
                import glob as _glob

                filelist = sorted(_glob.glob(os.path.join(folder, "**/*.abf"), recursive=True))
                dfs = []
                for f in filelist:
                    try:
                        result = preprocess_abf_subthreshold(f, protocol_filter or None, params)
                        dfs.append(result)
                    except Exception:
                        continue
                if dfs:
                    combined = pd.concat(dfs, sort=True)
                    tag = output_tag or str(int(time.time()))
                    save_subthres_data(combined, None, folder, tag)
                    return combined
        except Exception as exc:
            print(f"[Controller] Legacy batch also failed: {exc}")
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_module(self, name: str):
        """Look up a module from the analysis registry."""
        try:
            from pyAPisolation.analysis import get
            return get(name)
        except Exception:
            return None
