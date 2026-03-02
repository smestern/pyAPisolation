"""
Subthreshold analysis module — wraps legacy patch_subthres functions.

Extracts passive membrane properties (sag, time constant, voltage
deflection) from hyperpolarizing sweeps.
"""

import logging
import numpy as np

from ..base import AnalysisBase

logger = logging.getLogger(__name__)


class SubthresholdAnalysis(AnalysisBase):
    """
    Compute subthreshold membrane features per sweep.

    Wraps ``pyAPisolation.patch_subthres.subthres_a`` to extract sag ratio,
    membrane time constant (tau_m), and voltage deflection from
    hyperpolarizing current injections.

    Parameters
    ----------
    start : float
        Start time (s) of the analysis window.
    end : float
        End time (s) of the analysis window (0 = end of sweep).
    """

    name = "subthreshold"
    display_name = "Subthreshold Analysis"
    sweep_mode = "per_sweep"

    start: float = 0.0
    end: float = 0.0

    def analyze(self, x, y, c, **kwargs) -> dict:
        """
        Compute sag, tau_m, and voltage deflection for one sweep.

        Returns
        -------
        dict
            Keys: ``sag_ratio``, ``taum``, ``voltage_deflection``.
            Values are ``np.nan`` if the sweep has no hyperpolarizing
            current or the fit fails.
        """
        from ...patch_subthres import subthres_a

        end = self.end if self.end > 0 else x[-1]
        sag, taum, voltage_deflection = subthres_a(x, y, c, self.start, end)

        return {
            "sag_ratio": sag,
            "taum": taum,
            "voltage_deflection": voltage_deflection,
        }
