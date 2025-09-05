import numpy as np
import importlib.util

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False

from spikeinterface.core.node_pipeline import (
    PeakDetector,
)
from spikeinterface.core.recording_tools import get_noise_levels, get_channel_distances
from spikeinterface.sortingcomponents.detection.by_channel import ByChannelTorchPeakDetector, _torch_detect_peaks as _torch_detect_peaks

class LocallyExclusivePeakDetector(PeakDetector):
    """Detect peaks using the "locally exclusive" method."""

    name = "locally_exclusive"
    engine = "numba"
    preferred_mp_context = None
    # params_doc = (
    #     DetectPeakByChannel.params_doc
    #     + """
    # radius_um: float
    #     The radius to use to select neighbour channels for locally exclusive detection.
    # """
    # )

    def __init__(
        self,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        radius_um=50,
        noise_levels=None,
        random_chunk_kwargs={},
    ):
        if not HAVE_NUMBA:
            raise ModuleNotFoundError('"locally_exclusive" needs numba which is not installed')

        PeakDetector.__init__(self, recording, return_output=True)

        assert peak_sign in ("both", "neg", "pos")
        if noise_levels is None:
            self.noise_levels = get_noise_levels(recording, return_in_uV=False, **random_chunk_kwargs)
        else:
            self.noise_levels = noise_levels
        self.abs_thresholds = self.noise_levels * detect_threshold
        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        self.radius_um = radius_um
        self.detect_threshold = detect_threshold
        self.peak_sign = peak_sign
        # if remove_median:

        #     chunks = get_random_data_chunks(recording, return_in_uV=False, concatenated=True, **random_chunk_kwargs)
        #     medians = np.median(chunks, axis=0)
        #     medians = medians[None, :]
        #     print('medians', medians, noise_levels)
        # else:
        #     medians = None

        self.channel_distance = get_channel_distances(recording)
        self.neighbours_mask = self.channel_distance <= radius_um

    def get_trace_margin(self):
        return self.exclude_sweep_size

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        assert HAVE_NUMBA, "You need to install numba"

        # if medians is not None:
        #     traces = traces - medians

        traces_center = traces[self.exclude_sweep_size:-self.exclude_sweep_size, :]

        if self.peak_sign in ("pos", "both"):
            peak_mask = traces_center > self.abs_thresholds[None, :]
            peak_mask = _numba_detect_peak_pos(
                traces, traces_center, peak_mask, self.exclude_sweep_size, self.abs_thresholds, self.peak_sign, self.neighbours_mask
            )

        if self.peak_sign in ("neg", "both"):
            if self.peak_sign == "both":
                peak_mask_pos = peak_mask.copy()

            peak_mask = traces_center < -self.abs_thresholds[None, :]
            peak_mask = _numba_detect_peak_neg(
                traces, traces_center, peak_mask, self.exclude_sweep_size, self.abs_thresholds, self.peak_sign, self.neighbours_mask
            )

            if self.peak_sign == "both":
                peak_mask = peak_mask | peak_mask_pos

        # Find peaks and correct for time shift
        peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
        peak_sample_ind += self.exclude_sweep_size

        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]

        local_peaks = np.zeros(peak_sample_ind.size, dtype=self.get_dtype())
        local_peaks["sample_index"] = peak_sample_ind
        local_peaks["channel_index"] = peak_chan_ind
        local_peaks["amplitude"] = peak_amplitude
        local_peaks["segment_index"] = segment_index

        return (local_peaks,)

if HAVE_NUMBA:
    import numba

    @numba.jit(nopython=True, parallel=False)
    def _numba_detect_peak_pos(
        traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
    ):
        num_chans = traces_center.shape[1]
        for chan_ind in range(num_chans):
            for s in range(peak_mask.shape[0]):
                if not peak_mask[s, chan_ind]:
                    continue
                for neighbour in range(num_chans):
                    if not neighbours_mask[chan_ind, neighbour]:
                        continue
                    for i in range(exclude_sweep_size):
                        if chan_ind != neighbour:
                            peak_mask[s, chan_ind] &= traces_center[s, chan_ind] >= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] > traces[s + i, neighbour]
                        peak_mask[s, chan_ind] &= (
                            traces_center[s, chan_ind] >= traces[exclude_sweep_size + s + i + 1, neighbour]
                        )
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask

    @numba.jit(nopython=True, parallel=False)
    def _numba_detect_peak_neg(
        traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
    ):
        num_chans = traces_center.shape[1]
        for chan_ind in range(num_chans):
            for s in range(peak_mask.shape[0]):
                if not peak_mask[s, chan_ind]:
                    continue
                for neighbour in range(num_chans):
                    if not neighbours_mask[chan_ind, neighbour]:
                        continue
                    for i in range(exclude_sweep_size):
                        if chan_ind != neighbour:
                            peak_mask[s, chan_ind] &= traces_center[s, chan_ind] <= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] < traces[s + i, neighbour]
                        peak_mask[s, chan_ind] &= (
                            traces_center[s, chan_ind] <= traces[exclude_sweep_size + s + i + 1, neighbour]
                        )
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask
    
class LocallyExclusiveTorchPeakDetector(ByChannelTorchPeakDetector):
    """Detect peaks using the "locally exclusive" method with pytorch."""

    name = "locally_exclusive_torch"
    engine = "torch"
    preferred_mp_context = "spawn"
    # params_doc = (
    #     DetectPeakByChannel.params_doc
    #     + """
    # radius_um: float
    #     The radius to use to select neighbour channels for locally exclusive detection.
    # """
    # )

    def __init__(
        self,
        recording,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        noise_levels=None,
        device=None,
        radius_um=50,
        return_tensor=False,
        random_chunk_kwargs={},
    ):
        if not HAVE_TORCH:
            raise ModuleNotFoundError('"by_channel_torch" needs torch which is not installed')
        
        import torch
        
        ByChannelTorchPeakDetector.__init__(self, 
                                            recording, 
                                            recording,
                                            peak_sign,
                                            detect_threshold,
                                            exclude_sweep_ms,
                                            noise_levels,
                                            device,
                                            return_tensor,
                                            random_chunk_kwargs)

        self.channel_distance = get_channel_distances(recording)
        self.neighbour_indices_by_chan = []
        self.num_channels = recording.get_num_channels()
        for chan in range(self.num_channels):
            self.neighbour_indices_by_chan.append(np.nonzero(self.channel_distance[chan] <= radius_um)[0])
        self.max_neighbs = np.max([len(neigh) for neigh in self.neighbour_indices_by_chan])
        self.neighbours_idxs = self.num_channels * np.ones((self.num_channels, self.max_neighbs), dtype=int)
        for i, neigh in enumerate(self.neighbour_indices_by_chan):
            self.neighbours_idxs[i, : len(neigh)] = neigh

    def get_trace_margin(self):
        return self.exclude_sweep_size

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        peak_sample_ind, peak_chan_ind, peak_amplitude =  _torch_detect_peaks(
            traces, self.peak_sign, self.abs_thresholds, self.exclude_sweep_size, self.neighbor_idxs, self.device
        )
        if not self.return_tensor and isinstance(peak_sample_ind, torch.Tensor) and isinstance(peak_chan_ind, torch.Tensor):
            peak_sample_ind = np.array(peak_sample_ind.cpu())
            peak_chan_ind = np.array(peak_chan_ind.cpu())
            peak_amplitude = np.array(peak_amplitude.cpu())
            local_peaks = np.zeros(peak_sample_ind.size, dtype=self.get_dtype())
            local_peaks["sample_index"] = peak_sample_ind
            local_peaks["channel_index"] = peak_chan_ind
            local_peaks["amplitude"] = peak_amplitude
            local_peaks["segment_index"] = segment_index
        return (local_peaks, )