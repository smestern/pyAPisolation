"""
Example analysis module — a minimal template for new users.

Copy this file and modify ``analyze()`` to create your own analysis.
"""

import numpy as np

from ..base import AnalysisBase


class PeakDetector(AnalysisBase):
    """
    Minimal example: detect the peak voltage in each sweep.

    This shows the simplest possible analysis module.  You define
    parameters as class attributes and implement ``analyze()``.

    Usage
    -----
    ::

        from pyAPisolation.analysis import register
        from pyAPisolation.analysis.builtins.example import PeakDetector

        # Register so it's available by name
        register(PeakDetector)

        # Run on a single file
        module = PeakDetector(min_voltage=-30.0)
        result = module.run(file="path/to/recording.abf")

        # Get results as a DataFrame
        print(result.to_dataframe())
    """

    name = "peak_detector"
    display_name = "Peak Detector (Example)"
    sweep_mode = "per_sweep"

    # Parameters — just typed class attributes with defaults
    min_voltage: float = -20.0

    def analyze(self, x, y, c, **kwargs):
        """
        Find the peak voltage in a single sweep.

        Parameters
        ----------
        x : np.ndarray  — time array (1-D)
        y : np.ndarray  — voltage array (1-D)
        c : np.ndarray  — current array (1-D)

        Returns
        -------
        dict with keys ``peak_voltage``, ``peak_time``, ``peak_found``.
        """
        peak_v = float(np.max(y))
        peak_t = float(x[np.argmax(y)])

        if peak_v < self.min_voltage:
            return {"peak_found": False, "peak_voltage": peak_v, "peak_time": peak_t}

        return {"peak_found": True, "peak_voltage": peak_v, "peak_time": peak_t}
