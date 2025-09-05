from __future__ import annotations

from .iterative import IterativePeakDetector
from .matched_filtering import MatchedFilteringPeakDetector
from .by_channel import ByChannelPeakDetector
from .locally_exclusive import LocallyExclusivePeakDetector

detection_methods = {
    "locally_exclusive": LocallyExclusivePeakDetector,
    "matched_filering": MatchedFilteringPeakDetector,
    "iterative": IterativePeakDetector,
    "by_channel": ByChannelPeakDetector
}

