"""Trace-JSON helper shared by the analysis app and the legacy viewer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def load_trace_payload(file_path: str, max_points_per_sweep: int = 4000) -> Dict[str, Any]:
    """Load an ABF/NWB file and return a JSON-ready trace payload.

    Each sweep is downsampled to at most ``max_points_per_sweep`` points so
    payloads stay small enough for Plotly to render comfortably.

    Returns
    -------
    dict
        ``{ "file": <basename>, "protocol": <str>, "sweep_count": <int>,
            "time": [...], "sweeps": [{"index": i, "voltage": [...],
            "command": [...]}, ...] }``
    """

    from gigaseal.loadFile import loadFile  # local import keeps app boot light

    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(str(p))

    x, y, c = loadFile(str(p))
    x = np.asarray(x)
    y = np.asarray(y)
    c = np.asarray(c)

    if x.ndim == 1:
        x = x[None, :]
        y = y[None, :]
        c = c[None, :]

    n_sweeps, n_samples = y.shape
    stride = max(1, n_samples // max_points_per_sweep)

    time_axis = x[0, ::stride].tolist()
    sweeps = []
    for i in range(n_sweeps):
        sweeps.append(
            {
                "index": int(i),
                "voltage": y[i, ::stride].tolist(),
                "command": c[i, ::stride].tolist(),
            }
        )

    protocol = _peek_protocol(file_path)

    return {
        "file": p.name,
        "protocol": protocol,
        "sweep_count": int(n_sweeps),
        "samples_original": int(n_samples),
        "samples_returned": len(time_axis),
        "time": time_axis,
        "sweeps": sweeps,
    }


def _peek_protocol(file_path: str) -> Optional[str]:
    """Best-effort protocol name lookup; returns None on any failure."""

    try:
        import pyabf  # type: ignore

        if file_path.lower().endswith(".abf"):
            abf = pyabf.ABF(file_path, loadData=False)
            return getattr(abf, "protocol", None)
    except Exception:  # noqa: BLE001
        return None
    return None
