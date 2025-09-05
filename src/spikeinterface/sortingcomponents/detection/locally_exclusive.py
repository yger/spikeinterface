from .base import PeakDetectorWrapper

import importlib.util

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False

from spikeinterface.core.recording_tools import get_noise_levels, get_channel_distances

class LocallyExclusivePeakDetector(PeakDetectorWrapper):
    """Detect peaks using the "locally exclusive" method."""

    name = "locally_exclusive"
    engine = "numba"
    preferred_mp_context = None
    params_doc = (
        DetectPeakByChannel.params_doc
        + """
    radius_um: float
        The radius to use to select neighbour channels for locally exclusive detection.
    """
    )

    @classmethod
    def check_params(
        cls,
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

        # args = DetectPeakByChannel.check_params(
        #     recording,
        #     peak_sign=peak_sign,
        #     detect_threshold=detect_threshold,
        #     exclude_sweep_ms=exclude_sweep_ms,
        #     noise_levels=noise_levels,
        #     random_chunk_kwargs=random_chunk_kwargs,
        # )

        assert peak_sign in ("both", "neg", "pos")
        if noise_levels is None:
            noise_levels = get_noise_levels(recording, return_in_uV=False, **random_chunk_kwargs)
        abs_thresholds = noise_levels * detect_threshold
        exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)

        # if remove_median:

        #     chunks = get_random_data_chunks(recording, return_in_uV=False, concatenated=True, **random_chunk_kwargs)
        #     medians = np.median(chunks, axis=0)
        #     medians = medians[None, :]
        #     print('medians', medians, noise_levels)
        # else:
        #     medians = None

        channel_distance = get_channel_distances(recording)
        neighbours_mask = channel_distance <= radius_um
        return (peak_sign, abs_thresholds, exclude_sweep_size, neighbours_mask)

    @classmethod
    def get_method_margin(cls, *args):
        exclude_sweep_size = args[2]
        return exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, peak_sign, abs_thresholds, exclude_sweep_size, neighbours_mask):
        assert HAVE_NUMBA, "You need to install numba"

        # if medians is not None:
        #     traces = traces - medians

        traces_center = traces[exclude_sweep_size:-exclude_sweep_size, :]

        if peak_sign in ("pos", "both"):
            peak_mask = traces_center > abs_thresholds[None, :]
            peak_mask = _numba_detect_peak_pos(
                traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
            )

        if peak_sign in ("neg", "both"):
            if peak_sign == "both":
                peak_mask_pos = peak_mask.copy()

            peak_mask = traces_center < -abs_thresholds[None, :]
            peak_mask = _numba_detect_peak_neg(
                traces, traces_center, peak_mask, exclude_sweep_size, abs_thresholds, peak_sign, neighbours_mask
            )

            if peak_sign == "both":
                peak_mask = peak_mask | peak_mask_pos

        # Find peaks and correct for time shift
        peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
        peak_sample_ind += exclude_sweep_size

        return peak_sample_ind, peak_chan_ind

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