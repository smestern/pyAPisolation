"""
Batch runner for analysis modules.

Provides :func:`run_batch` to process a folder (or list) of ABF files
through any registered analysis module, with optional protocol filtering
and multiprocessing.
"""

import os
import glob
import logging
from typing import List, Optional, Union
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from .base import AnalysisBase
from .result import AnalysisResult

logger = logging.getLogger(__name__)


# ======================================================================
# Public API
# ======================================================================

def run_batch(
    module: AnalysisBase,
    files_or_folder: Union[str, List[str]],
    protocol_filter: Optional[str] = None,
    selected_sweeps: Optional[List[int]] = None,
    n_jobs: int = 1,
    progress_callback=None,
    **kwargs,
) -> AnalysisResult:
    """
    Run *module* over every ABF file in *files_or_folder*.

    Parameters
    ----------
    module : AnalysisBase
        The analysis module to run.
    files_or_folder : str or list[str]
        A folder path (will be globbed for ``*.abf`` recursively),
        or an explicit list of file paths.
    protocol_filter : str, optional
        Only process files whose ``cellData.protocol`` contains this
        substring (case-insensitive).
    selected_sweeps : list[int], optional
        Restrict to these sweep indices.  ``None`` → all sweeps.
    n_jobs : int
        Number of parallel workers.  ``1`` → sequential (default).
    progress_callback : callable, optional
        ``callback(completed: int, total: int)`` invoked after each file.
        Useful for GUI progress bars.  ``None`` → no-op.

    Returns
    -------
    AnalysisResult
        A combined result whose ``.to_dataframe()`` contains rows from
        every processed file.
    """
    # --- resolve file list ------------------------------------------------
    if isinstance(files_or_folder, str):
        if os.path.isdir(files_or_folder):
            filelist = sorted(
                glob.glob(os.path.join(files_or_folder, "**/*.abf"),
                           recursive=True)
            )
        else:
            filelist = sorted(glob.glob(files_or_folder))
    elif isinstance(files_or_folder, list):
        filelist = list(files_or_folder)
    else:
        raise TypeError(
            f"files_or_folder must be str or list, got {type(files_or_folder)}"
        )

    if not filelist:
        return AnalysisResult(
            name=module.name, file_path="none", success=False,
            errors=[f"No files found in {files_or_folder}"]
        )

    logger.info(f"Batch run: {len(filelist)} files with module '{module.name}'")

    # --- optional protocol filter -----------------------------------------
    if protocol_filter:
        filelist = _filter_by_protocol(filelist, protocol_filter)
        if not filelist:
            return AnalysisResult(
                name=module.name, file_path="none", success=False,
                errors=[f"No files matched protocol filter '{protocol_filter}'"]
            )
        logger.info(f"After protocol filter: {len(filelist)} files")

    # --- run analysis -----------------------------------------------------
    results: List[AnalysisResult] = []

    if n_jobs > 1 and len(filelist) > 1:
        # Parallel execution
        # Note: module.analyze must be picklable; for complex modules
        # that import heavy GUI deps this may need adjusting.
        with ProcessPoolExecutor(max_workers=n_jobs) as pool:
            futures = {
                pool.submit(_run_one, module, f, selected_sweeps, kwargs): f
                for f in filelist
            }
            for i, future in enumerate(futures):
                try:
                    results.append(future.result())
                except Exception as exc:
                    fp = futures[future]
                    logger.error(f"Failed on {fp}: {exc}")
                    results.append(AnalysisResult(
                        name=module.name, file_path=fp, success=False,
                        errors=[str(exc)]
                    ))
                if progress_callback is not None:
                    progress_callback(i + 1, len(filelist))
    else:
        # Sequential execution
        for i, filepath in enumerate(filelist):
            try:
                res = module.run(file=filepath, selected_sweeps=selected_sweeps,
                                 **kwargs)
                results.append(res)
            except Exception as exc:
                logger.error(f"Failed on {filepath}: {exc}")
                results.append(AnalysisResult(
                    name=module.name, file_path=filepath, success=False,
                    errors=[str(exc)]
                ))
            if progress_callback is not None:
                progress_callback(i + 1, len(filelist))

    # --- combine results --------------------------------------------------
    combined = AnalysisResult.concatenate(results)
    logger.info(
        f"Batch complete: {sum(r.success for r in results)}/{len(results)} "
        f"files succeeded"
    )
    return combined


# ======================================================================
# I/O helpers
# ======================================================================

def save_results(
    result: AnalysisResult,
    output_dir: str,
    tag: str = "",
    fmt: str = "csv",
) -> str:
    """
    Save an AnalysisResult's DataFrame to disk.

    Parameters
    ----------
    result : AnalysisResult
    output_dir : str
        Directory to write into (created if needed).
    tag : str
        Optional suffix appended to the filename.
    fmt : str
        ``"csv"`` or ``"xlsx"``.

    Returns
    -------
    str
        Path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = result.to_dataframe()

    suffix = f"_{tag}" if tag else ""
    basename = f"{result.name}{suffix}"

    if fmt == "xlsx":
        path = os.path.join(output_dir, f"{basename}.xlsx")
        df.to_excel(path, index=False)
    else:
        path = os.path.join(output_dir, f"{basename}.csv")
        df.to_csv(path, index=False)

    logger.info(f"Saved results to {path}")
    return path


# ======================================================================
# Internal helpers
# ======================================================================

def _run_one(module, filepath, selected_sweeps, extra_kwargs):
    """Wrapper for ProcessPoolExecutor (must be top-level for pickling)."""
    return module.run(file=filepath, selected_sweeps=selected_sweeps,
                      **extra_kwargs)


def _filter_by_protocol(filelist, protocol_filter):
    """Keep only files whose protocol matches the filter substring."""
    from ..dataset import cellData

    filtered = []
    pf_lower = protocol_filter.lower()
    for fp in filelist:
        try:
            data = cellData(fp)
            if pf_lower in data.protocol.lower():
                filtered.append(fp)
        except Exception:
            logger.debug(f"Could not load {fp} for protocol check, skipping")
    return filtered
