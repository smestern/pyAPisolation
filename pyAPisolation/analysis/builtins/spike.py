"""
Spike analysis module — wraps the legacy featureExtractor.

Demonstrates how to build a per-sweep analysis module that delegates
to existing, well-tested code.
"""

import copy
import logging
import numpy as np
import pandas as pd

from ..base import AnalysisBase

logger = logging.getLogger(__name__)


class SpikeAnalysis(AnalysisBase):
    """
    Detect action potentials and extract spike features using the ipfx
    feature extractor.

    This module wraps ``pyAPisolation.featureExtractor.analyze_sweep``
    so all existing spike-detection logic is reused.

    Parameters
    ----------
    dv_cutoff : float
        Minimum dV/dt (mV/ms) to consider a spike.
    start : float
        Time (s) to start looking for spikes.
    end : float
        Time (s) to stop looking (0 = end of sweep).
    max_interval : float
        Max interval (s) between consecutive spike peaks.
    min_height : float
        Min spike height (mV) from threshold to peak.
    min_peak : float
        Min absolute voltage (mV) at spike peak.
    thresh_frac : float
        Fraction of spike height for threshold.
    filter : int
        Lowpass filter frequency (Hz) for ipfx (0 = off).
    bessel_filter : int
        Bessel filter cutoff (Hz). -1 = off.
    """

    name = "spike"
    display_name = "Spike Analysis"
    sweep_mode = "per_sweep"

    # Parameters as typed class attributes with defaults
    dv_cutoff: float = 7.0
    start: float = 0.0
    end: float = 0.0
    max_interval: float = 0.005
    min_height: float = 2.0
    min_peak: float = -10.0
    thresh_frac: float = 0.2
    filter: int = 0
    bessel_filter: int = -1

    def analyze(self, x, y, c, **kwargs) -> dict:
        """
        Run ipfx spike extraction on a single sweep.

        Returns a flat dict of spike features.  If no spikes are found,
        returns ``{"spike_count": 0}``.
        """
        from ...featureExtractor import analyze_sweep

        # Build ipfx-compatible param_dict from our class attributes
        param_dict = {
            "dv_cutoff": self.dv_cutoff,
            "start": self.start,
            "end": self.end,
            "max_interval": self.max_interval,
            "min_height": self.min_height,
            "min_peak": self.min_peak,
            "thresh_frac": self.thresh_frac,
            "filter": self.filter,
        }

        bessel = self.bessel_filter if self.bessel_filter != -1 else None

        if param_dict["end"] == 0.0:
            param_dict["end"] = x[-1]

        spike_df, spike_train = analyze_sweep(x, y, c,
                                               param_dict=param_dict,
                                               bessel_filter=bessel)

        # Build output dict
        result = {"spike_count": len(spike_df)}

        if not spike_df.empty:
            # Include per-spike features from ipfx
            for col in spike_df.columns:
                values = spike_df[col].tolist()
                result[col] = values if len(values) > 1 else values[0]

        # Include spike-train-level features
        if isinstance(spike_train, dict):
            for key, val in spike_train.items():
                result[f"train_{key}"] = val

        return result
