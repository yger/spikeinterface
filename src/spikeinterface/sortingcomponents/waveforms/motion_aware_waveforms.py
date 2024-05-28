from __future__ import annotations

from pathlib import Path
import json
from typing import List, Optional
import scipy.signal
import numpy as np

from spikeinterface.core import BaseRecording
from spikeinterface.core.node_pipeline import PipelineNode, WaveformsNode, find_parent_of_type
from spikeinterface.core import get_channel_distances

class MotionAwareWaveforms(WaveformsNode):
    """
    Waveform Denoiser based on a 

    Parameters
    ----------
    recording: BaseRecording
        The recording extractor object
    return_output: bool, default: True
        Whether to return output from this node
    parents: list of PipelineNodes, default: None
        The parent nodes of this node
    motion:

    """

    def __init__(
        self,
        recording: BaseRecording,
        motion: np.array = None,
        return_output: bool = True,
        radius_um: float=100, 
        parents: Optional[List[PipelineNode]] = None,
        sparse=False,
    ):
        waveform_extractor = find_parent_of_type(parents, WaveformsNode)
        if waveform_extractor is None:
            raise TypeError(f"SavGolDenoiser should have a single {WaveformsNode.__name__} in its parents")

        super().__init__(
            recording,
            waveform_extractor.ms_before,
            waveform_extractor.ms_after,
            return_output=return_output,
            parents=parents,
        )

        self.channel_locations = self.recording.get_channel_locations()
        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um
        self.motion = motion
        self.sparse = sparse
        self.fs = recording.sampling_frequency
        self._kwargs.update(dict(motion=motion, fs=self.fs, radius_um=radius_um, sparse=sparse))

    def compute(self, traces, peaks, waveforms):
        # interpolate given some motion
        from spikeinterface.generation.drift_tools import move_dense_templates
        from spikeinterface.sortingcomponents.motion_interpolation import _get_closest_ind

        peak_times = peaks['sample_index'] / self.fs
        motion_corrected_waveforms = np.zeros(waveforms.shape, dtype=waveforms.dtype)

        bin_inds = _get_closest_ind(self.motion['temporal_bins'], peak_times)
        for bin_ind in np.unique(bin_inds):
            # Step 1 : channel motion
            if self.motion['spatial_bins'].shape[0] == 1:
                # rigid motion : same motion for all channels
                channel_motions = self.motion['motion'][bin_ind, 0]
            else:
                # non rigid : interpolation channel motion for this temporal bin
                f = scipy.interpolate.interp1d(
                    self.motion['spatial_bins'], self.motion['motion'][bin_ind, :], kind="linear", 
                        axis=0, 
                        bounds_error=False, 
                        fill_value="extrapolate"
                )
                locs = self.channel_locations[:, 1]
                channel_motions = f(locs)
            
            mask = bin_inds == bin_ind
            displacements = np.zeros((np.sum(mask), 2), dtype=np.float32)
            local_motions = channel_motions[peaks[mask]['channel_index']]
            displacements[:, 1] = local_motions
            
            for count in np.nonzero(mask)[0]:
                main_chan = peaks[count]["channel_index"]
                (chan_inds,) = np.nonzero(self.neighbours_mask[main_chan])

                #f chan_inds is not None:
                #    channel_locations_moved = channel_locations_moved[chan_inds]

                if self.sparse:
                    pass
                else:
                    source_probe = self.recording.get_probe()
                    dest_probe = self.recording.get_probe()
                    a = move_dense_templates(waveforms[count][np.newaxis, :, :], displacements[count][np.newaxis, :], source_probe, dest_probe)
                #motion_corrected_waveforms[count] = a

        return motion_corrected_waveforms
