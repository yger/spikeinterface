from __future__ import annotations

from pathlib import Path
import json
from typing import List, Optional
import numpy as np


from spikeinterface.core import BaseRecording
from spikeinterface.core.node_pipeline import PipelineNode, WaveformsNode, find_parent_of_type


class TemporalSlicer(WaveformsNode):
    """
    Hanning Filter to keep only the energy at the center of the waveforms

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor object
    return_output: bool, default: True
        Whether to return output from this node
    parents: list of PipelineNodes, default: None
        The parent nodes of this node
    order: float, default: 1
        the exponent for the Hanning window
    """

    def __init__(
        self,
        recording: BaseRecording,
        return_output: bool = True,
        parents: Optional[List[PipelineNode]] = None,
        ms_before: float = 0.5,
        ms_after: float = 0.5
    ):
        waveform_extractor = find_parent_of_type(parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"HanningFilter should have a single {WaveformsNode.__name__} in its parents")

        assert ms_before < waveform_extractor.ms_before
        assert ms_after < waveform_extractor.ms_after


        super().__init__(
            recording,
            ms_before,
            ms_after,
            return_output=return_output,
            parents=parents,
        )

        self.center = waveform_extractor.nbefore
        self._kwargs.update(dict(center=self.center))

    def compute(self, traces, peaks, waveforms):
        # Slice waveforms
        sliced_waveforms = waveforms[:, self.center-self.nbefore:self.center+self.nafter, :]
        return sliced_waveforms
