"""Background analysis job runner for the web app.

Wraps :func:`gigaseal.analysis.runner.run_batch` in a thread pool. Each job
holds an in-memory :class:`Job` record plus an on-disk parquet/csv export.

A global semaphore caps concurrent jobs per ``WebConfig.max_concurrent_jobs``;
overflow is reported back as HTTP 429 by the route layer.
"""

from __future__ import annotations

import logging
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import math

import numpy as np

from .config import WebConfig

logger = logging.getLogger(__name__)


def _json_safe(value: Any) -> Any:
    """Recursively coerce NaN / ±inf / numpy scalars into JSON-friendly values.

    DataFrame cells frequently hold nested dicts or numpy arrays whose inner
    NaN / Infinity tokens slip past pandas' top-level ``where(notna(), None)``
    sanitization and break ``JSON.parse`` in the browser.
    """
    if value is None:
        return None
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    try:
        import pandas as pd

        if isinstance(value, pd.Series):
            return [_json_safe(v) for v in value.tolist()]
        if isinstance(value, pd.DataFrame):
            return [_json_safe(r) for r in value.to_dict(orient="records")]
        if pd.isna(value):
            return None
    except Exception:  # noqa: BLE001
        pass
    return str(value)


@dataclass
class Job:
    job_id: str
    session_id: str
    module_name: str
    file_paths: List[str]
    params: Dict[str, Any]
    selected_sweeps: Optional[List[int]]
    status: str = "queued"  # queued | running | done | error | rejected
    progress: float = 0.0  # 0.0–1.0
    completed: int = 0
    total: int = 0
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None
    result_path: Optional[Path] = None  # parquet/csv on disk
    columns: List[str] = field(default_factory=list)
    row_count: int = 0
    preview_rows: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "module": self.module_name,
            "status": self.status,
            "progress": round(self.progress, 3),
            "completed": self.completed,
            "total": self.total,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
            "row_count": self.row_count,
            "columns": self.columns,
            "preview": self.preview_rows,
        }


class JobManager:
    """Thread-pooled job manager with a global concurrency cap."""

    def __init__(self, config: WebConfig):
        self.config = config
        self._jobs: Dict[str, Job] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.RLock()
        # Use a thread pool — ProcessPoolExecutor is harder to embed
        # in Flask workers and the inner run_batch already supports
        # n_jobs>1 for CPU-bound work when desired.
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, config.max_concurrent_jobs),
            thread_name_prefix="gigaseal-job",
        )

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        session_id: str,
        module_name: str,
        file_paths: List[str],
        params: Dict[str, Any],
        selected_sweeps: Optional[List[int]],
        result_dir: Path,
    ) -> Job:
        with self._lock:
            active = sum(
                1 for j in self._jobs.values() if j.status in ("queued", "running")
            )
            if active >= self.config.max_concurrent_jobs:
                raise JobRejected(
                    f"server is busy ({active}/{self.config.max_concurrent_jobs} "
                    "concurrent jobs); please retry shortly"
                )
            job = Job(
                job_id=uuid.uuid4().hex[:12],
                session_id=session_id,
                module_name=module_name,
                file_paths=list(file_paths),
                params=dict(params),
                selected_sweeps=selected_sweeps,
                total=len(file_paths),
            )
            self._jobs[job.job_id] = job

        result_dir.mkdir(parents=True, exist_ok=True)
        future = self._executor.submit(
            self._run, job, result_dir
        )
        with self._lock:
            self._futures[job.job_id] = future
        return job

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, session_id: str, job_id: str) -> Optional[Job]:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None or job.session_id != session_id:
            return None
        return job

    def list_for_session(self, session_id: str) -> List[Job]:
        with self._lock:
            return [j for j in self._jobs.values() if j.session_id == session_id]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _run(self, job: Job, result_dir: Path) -> None:
        # Local imports keep test-time import light and avoid pulling
        # IPFX during app construction.
        from gigaseal.analysis import get as get_module, run_batch
        from gigaseal.analysis.result import AnalysisResult

        job.status = "running"
        job.started_at = time.time()

        try:
            module = get_module(job.module_name)
            if module is None:
                raise RuntimeError(f"unknown analysis module: {job.module_name}")
            # Fresh instance with this job's parameter overrides.
            module = type(module)(**job.params)

            def on_progress(done: int, total: int) -> None:
                job.completed = done
                job.total = total
                job.progress = (done / total) if total else 0.0

            result: AnalysisResult = run_batch(
                module,
                job.file_paths,
                selected_sweeps=job.selected_sweeps,
                n_jobs=1,
                progress_callback=on_progress,
            )

            df = result.to_dataframe()
            csv_path = result_dir / f"{job.job_id}.csv"
            df.to_csv(csv_path, index=False)
            job.result_path = csv_path
            job.columns = list(df.columns)
            job.row_count = int(len(df))
            # Replace NaN *and* ±inf with None so the JSON response stays
            # spec-compliant; the browser's JSON.parse rejects literal NaN /
            # Infinity tokens and the poller would otherwise see an empty
            # body and never observe status == "done".
            preview = df.head(50).replace([np.inf, -np.inf], np.nan)
            raw_rows = preview.where(preview.notna(), None).to_dict(
                orient="records"
            )
            # DataFrame cells can themselves contain dicts/arrays whose
            # nested NaNs survive the top-level sanitization; walk them.
            job.preview_rows = [_json_safe(row) for row in raw_rows]
            job.progress = 1.0
            job.status = "done"
        except Exception as exc:  # noqa: BLE001
            logger.exception("job %s failed", job.job_id)
            job.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            job.status = "error"
        finally:
            job.finished_at = time.time()


class JobRejected(Exception):
    """Raised when the global concurrency cap blocks a new submission."""
