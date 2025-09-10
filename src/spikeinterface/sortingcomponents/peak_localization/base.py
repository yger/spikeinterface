from spikeinterface.core.node_pipeline import (
    PipelineNode,
)

from spikeinterface.core import get_channel_distances

class LocalizeBase(PipelineNode):
    def __init__(self, recording, return_output=True, parents=None, radius_um=75.0):
        PipelineNode.__init__(self, recording, return_output=return_output, parents=parents)

        self.radius_um = radius_um
        self.contact_locations = recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um
        self._kwargs["radius_um"] = radius_um

    def get_dtype(self):
        return self._dtype