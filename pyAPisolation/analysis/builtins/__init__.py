"""
Built-in analysis modules.

Importing this package auto-registers the standard analysis modules
(spike, subthreshold, peak_detector example).
"""

from ..registry import register

from .spike import SpikeAnalysis
from .subthreshold import SubthresholdAnalysis
from .example import PeakDetector

# Register all built-in modules
register(SpikeAnalysis)
register(SubthresholdAnalysis)
register(PeakDetector)
