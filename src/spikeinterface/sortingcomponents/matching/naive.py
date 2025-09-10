"""Sorting components: template matching."""

from __future__ import annotations


import numpy as np
from spikeinterface.core import get_noise_levels, get_channel_distances
from spikeinterface.sortingcomponents.peak_detection.locally_exclusive import LocallyExclusivePeakDetector


from .base import BaseTemplateMatching, _base_matching_dtype


class NaiveMatching(BaseTemplateMatching):

    name = "naive"
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
        templates=None,
        peak_sign="neg",
        exclude_sweep_ms=0.1,
        detect_threshold=5,
        noise_levels=None,
        radius_um=100.0,
        random_chunk_kwargs={},
    ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=True, parents=None)

        self.templates_array = self.templates.get_dense_templates()

        if noise_levels is None:
            self.noise_levels = get_noise_levels(recording, **random_chunk_kwargs, return_in_uV=False)
        else:
            self.noise_levels = noise_levels
        self.detect_threshold = detect_threshold
        self.abs_thresholds = self.noise_levels * self.detect_threshold
        self.peak_sign = peak_sign
        self.exclude_sweep_ms = exclude_sweep_ms
        self.nbefore = self.templates.nbefore
        self.nafter = self.templates.nafter
        self.radius_um = radius_um
        self.margin = max(self.nbefore, self.nafter)
        self.peak_detector = LocallyExclusivePeakDetector(
            recording,
            peak_sign=self.peak_sign,
            detect_threshold=self.detect_threshold,
            noise_levels=self.noise_levels,
            radius_um = self.radius_um,
            exclude_sweep_ms=self.exclude_sweep_ms,
        )

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):

        if self.margin > 0:
            peak_traces = traces[self.margin : -self.margin, :]
        else:
            peak_traces = traces

        (local_peaks, ) = self.peak_detector.compute(
            peak_traces, start_frame, end_frame, segment_index, self.margin
        )

        peak_sample_ind = local_peaks["sample_index"]
        peak_chan_ind = local_peaks["channel_index"]
        peak_sample_ind += self.margin

        spikes = np.zeros(peak_sample_ind.size, dtype=_base_matching_dtype)
        spikes["sample_index"] = peak_sample_ind
        spikes["channel_index"] = peak_chan_ind

        # naively take the closest template
        for i in range(peak_sample_ind.size):
            i0 = peak_sample_ind[i] - self.nbefore
            i1 = peak_sample_ind[i] + self.nafter

            waveforms = traces[i0:i1, :]
            dist = np.sum(np.sum((self.templates_array - waveforms[None, :, :]) ** 2, axis=1), axis=1)
            cluster_index = np.argmin(dist)

            spikes["cluster_index"][i] = cluster_index
            spikes["amplitude"][i] = 1.0

        return spikes
