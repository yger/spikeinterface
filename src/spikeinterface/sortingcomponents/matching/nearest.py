"""Sorting components: template matching."""

from __future__ import annotations


import numpy as np
from spikeinterface.core import get_noise_levels, get_channel_distances


from .base import BaseTemplateMatching, _base_matching_dtype


class NearestTemplatesPeeler(BaseTemplateMatching):

    name = "nearest"
    need_noise_levels = True
    params_doc = """
    peak_sign : 'neg' | 'pos' | 'both'
        The peak sign to use for detection
    exclude_sweep_ms : float
        The exclusion window (in ms) around a detected peak to exclude other peaks on neighboring channels
    detect_threshold : float
        The threshold for peak detection in term of k x MAD
    noise_levels : None | array
        If None the noise levels are estimated using random chunks of the recording. If array it should be an array of size (num_channels,) with the noise level of each channel
    radius_um : float
        The radius to define the neighborhood between channels in micrometers
    random_chunk_kwargs : dict
        The kwargs for get_noise_levels if noise_levels is None
    """

    def __init__(
        self,
        recording,
        templates,
        return_output=True,
        peak_sign="neg",
        exclude_sweep_ms=0.1,
        detect_threshold=5,
        noise_levels=None,
        radius_um=100.0,
        random_chunk_kwargs={},
    ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=return_output)

        self.templates_array = self.templates.get_dense_templates()

        if noise_levels is None:
            self.noise_levels = get_noise_levels(recording, **random_chunk_kwargs, return_in_uV=False)
        else:
            self.noise_levels = noise_levels
        self.abs_threholds = self.noise_levels * detect_threshold
        self.peak_sign = peak_sign
        channel_distance = get_channel_distances(recording)
        self.neighbours_mask = channel_distance <= radius_um
        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        self.nbefore = self.templates.nbefore
        self.nafter = self.templates.nafter
        self.margin = max(self.nbefore, self.nafter)

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):
        from spikeinterface.sortingcomponents.peak_detection.locally_exclusive import (
            detect_peaks_numba_locally_exclusive_on_chunk,
        )
        from scipy.spatial.distance import cdist

        if self.margin > 0:
            peak_traces = traces[self.margin : -self.margin, :]
        else:
            peak_traces = traces
        peak_sample_ind, peak_chan_ind = detect_peaks_numba_locally_exclusive_on_chunk(
            peak_traces, self.peak_sign, self.abs_threholds, self.exclude_sweep_size, self.neighbours_mask
        )
        peak_sample_ind += self.margin

        spikes = np.empty(peak_sample_ind.size, dtype=_base_matching_dtype)
        spikes["sample_index"] = peak_sample_ind
        spikes["channel_index"] = peak_chan_ind

        waveforms = traces[spikes["sample_index"][:, None] + np.arange(-self.nbefore, self.nafter)]
        num_templates = len(self.templates_array)
        XA = self.templates_array.reshape(num_templates,-1)

        # naively take the closest template
        for main_chan in np.unique(spikes["channel_index"]):
            (idx,) = np.nonzero(spikes["channel_index"] == main_chan)
            XB = waveforms[idx].reshape(len(idx),-1)
            dist = cdist(XA, XB, "euclidean")
            cluster_index = np.argmin(dist, 0)
            spikes["cluster_index"][idx] = cluster_index
            spikes["amplitude"][idx] = 1.0

        return spikes


class NearestTemplatesSVDPeeler(BaseTemplateMatching):

    name = "nearest_svd"
    need_noise_levels = True
    params_doc = """
    svd_model : The svd model used to project the waveforms
    peak_sign : 'neg' | 'pos' | 'both'
        The peak sign to use for detection
    exclude_sweep_ms : float
        The exclusion window (in ms) around a detected peak to exclude other peaks on neighboring channels
    detect_threshold : float
        The threshold for peak detection in term of k x MAD
    noise_levels : None | array
        If None the noise levels are estimated using random chunks of the recording. If array it should be an array of size (num_channels,) with the noise level of each channel
    radius_um : float
        The radius to define the neighborhood between channels in micrometers
    random_chunk_kwargs : dict
        The kwargs for get_noise_levels if noise_levels is None
    """

    def __init__(
        self,
        recording,
        templates,
        svd_model,
        return_output=True,
        peak_sign="neg",
        exclude_sweep_ms=0.1,
        detect_threshold=5,
        noise_levels=None,
        radius_um=100.0,
        random_chunk_kwargs={},
    ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=return_output)

        self.templates_array = self.templates.get_dense_templates()

        if noise_levels is None:
            self.noise_levels = get_noise_levels(recording, **random_chunk_kwargs, return_in_uV=False)
        else:
            self.noise_levels = noise_levels
        self.abs_threholds = self.noise_levels * detect_threshold
        self.peak_sign = peak_sign
        channel_distance = get_channel_distances(recording)
        self.neighbours_mask = channel_distance <= radius_um
        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        self.nbefore = self.templates.nbefore
        self.nafter = self.templates.nafter
        self.margin = max(self.nbefore, self.nafter)
        
        templates_array = templates.get_dense_templates()
        n_templates = templates_array.shape[0]
        self.num_channels = recording.get_num_channels()
        self.svd_model = svd_model
        self.svd_templates = np.zeros((n_templates, self.svd_model.n_components, self.num_channels), "float32")
        for i in range(n_templates):
            self.svd_templates[i] = self.svd_model.transform(templates_array[i].T).T

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):
        from spikeinterface.sortingcomponents.peak_detection.locally_exclusive import (
            detect_peaks_numba_locally_exclusive_on_chunk,
        )
        from scipy.spatial.distance import cdist

        if self.margin > 0:
            peak_traces = traces[self.margin : -self.margin, :]
        else:
            peak_traces = traces
        peak_sample_ind, peak_chan_ind = detect_peaks_numba_locally_exclusive_on_chunk(
            peak_traces, self.peak_sign, self.abs_threholds, self.exclude_sweep_size, self.neighbours_mask
        )
        peak_sample_ind += self.margin

        spikes = np.empty(peak_sample_ind.size, dtype=_base_matching_dtype)
        spikes["sample_index"] = peak_sample_ind
        spikes["channel_index"] = peak_chan_ind
        spikes["amplitudes"] = 1


        temporal_waveforms = to_temporal_representation(waveforms)
        projected_temporal_waveforms = self.pca_model.transform(temporal_waveforms)
        projected_waveforms = from_temporal_representation(projected_temporal_waveforms, self.num_channels)

        for main_chan in np.unique(spikes["channel_index"]):
            (idx,) = np.nonzero(spikes["channel_index"] == main_chan)
            (chan_inds,) = np.nonzero(self.sparsity_mask[main_chan])
            local_svds = projected_waveforms[idx][:, :, : len(chan_inds)]
            XA = local_svds.reshape(local_svds.shape[0], -1)
            XB = self.svd_templates[:, :, chan_inds].reshape(self.svd_templates.shape[0], -1)
            distances = cdist(XA, XB, metric="euclidean")
            spikes["cluster_index"][idx] = np.argmin(distances, axis=1)
            

        return spikes
