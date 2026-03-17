"""
Built-in analysis modules.

Importing this package auto-registers the standard analysis modules
(spike, subthreshold, peak_detector example).
"""

from ..registry import register

from .spike import SpikeAnalysis, LegacySpikeAnalysis
from .subthreshold import SubthresholdAnalysis
from .example import PeakDetector

# Register all built-in modules
register(SpikeAnalysis)
register(SubthresholdAnalysis)
register(LegacySpikeAnalysis)

#you could register example here as well, but we don't want it to show up in the GUI by default since it's just a demo
#register(PeakDetector)