"""Sorting components: peak localization."""

from __future__ import annotations
import numpy as np

from spikeinterface.postprocessing.unit_locations import dtype_localize_by_method

from spikeinterface.core.node_pipeline import PipelineNode

class LocalizePeakChannel(PipelineNode):
    """Localize peaks using the channel"""

    name = "peak_channel"
    params_doc = """
    """

    def __init__(self, recording, parents=None, return_output=True):
        PipelineNode.__init__(self, recording, return_output, parents=parents)
        self._dtype = np.dtype(dtype_localize_by_method["center_of_mass"])

        self.contact_locations = recording.get_channel_locations()

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, peaks):
        peak_locations = np.zeros(peaks.size, dtype=self._dtype)

        for index, main_chan in enumerate(peaks["channel_index"]):
            locations = self.contact_locations[main_chan, :]
            peak_locations["x"][index] = locations[0]
            peak_locations["y"][index] = locations[1]

        return peak_locations