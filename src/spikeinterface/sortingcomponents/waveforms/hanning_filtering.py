from __future__ import annotations

from pathlib import Path
import json
from typing import List, Optional
import numpy as np


from spikeinterface.core import BaseRecording
from spikeinterface.core.node_pipeline import PipelineNode, WaveformsNode, find_parent_of_type


class HanningFilter(WaveformsNode):
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
        order: int = 1,
    ):
        waveform_extractor = find_parent_of_type(parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"HanningFilter should have a single {WaveformsNode.__name__} in its parents")

        super().__init__(
            recording,
            waveform_extractor.ms_before,
            waveform_extractor.ms_after,
            return_output=return_output,
            parents=parents,
        )

        self.order = order
        self.hanning = np.hanning(waveform_extractor.nbefore + waveform_extractor.nafter)**self.order
        self.hanning = self.hanning[:, None]

        self._kwargs.update(dict(order=order, hanning=self.hanning))

    def compute(self, traces, peaks, waveforms):
        # Denoise
        denoised_waveforms = waveforms * self.hanning

        return denoised_waveforms
