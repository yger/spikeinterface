"""Sorting components: peak detection."""

from __future__ import annotations


import copy
from typing import Tuple, List, Optional
import importlib.util

import numpy as np

from spikeinterface.core.job_tools import (
    _shared_job_kwargs_doc,
    split_job_kwargs,
)
from spikeinterface.core.recording_tools import get_noise_levels, get_channel_distances, get_random_data_chunks

from spikeinterface.core.baserecording import BaseRecording
from spikeinterface.core.node_pipeline import (
    PeakDetector,
    WaveformsNode,
    ExtractSparseWaveforms,
    run_node_pipeline,
    base_peak_dtype,
)

from spikeinterface.postprocessing.localization_tools import get_convolution_weights

from .tools import make_multi_method_doc

numba_spec = importlib.util.find_spec("numba")
if numba_spec is not None:
    HAVE_NUMBA = True
else:
    HAVE_NUMBA = False

torch_spec = importlib.util.find_spec("torch")
if torch_spec is not None:
    torch_nn_functional_spec = importlib.util.find_spec("torch.nn")
    if torch_nn_functional_spec is not None:
        HAVE_TORCH = True
    else:
        HAVE_TORCH = False
else:
    HAVE_TORCH = False

"""
TODO:
    * remove the wrapper class and move  all implementation to instance
"""

expanded_base_peak_dtype = np.dtype(base_peak_dtype + [("iteration", "int8")])













class DetectPeakLocallyExclusive(PeakDetectorWrapper):
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


class DetectPeakMatchedFiltering(PeakDetector):
    """Detect peaks using the 'matched_filtering' method."""

    name = "matched_filtering"
    engine = "numba"
    preferred_mp_context = None
    params_doc = (
        DetectPeakByChannel.params_doc
        + """
    radius_um : float
        The radius to use to select neighbour channels for locally exclusive detection.
    prototype : array
        The canonical waveform of action potentials
    ms_before : float
        The time in ms before the maximial value of the absolute prototype
    weight_method : dict
        Parameter that should be provided to the get_convolution_weights() function
        in order to know how to estimate the positions. One argument is mode that could
        be either gaussian_2d (KS like) or exponential_3d (default)
    """
    )

    def __init__(
        self,
        recording,
        prototype,
        ms_before,
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        radius_um=50,
        noise_levels=None,
        random_chunk_kwargs={"num_chunks_per_segment": 5},
        weight_method={},
    ):
        PeakDetector.__init__(self, recording, return_output=True)
        from scipy.sparse import csr_matrix

        if not HAVE_NUMBA:
            raise ModuleNotFoundError('matched_filtering" needs numba which is not installed')

        self.exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        channel_distance = get_channel_distances(recording)
        self.neighbours_mask = channel_distance <= radius_um

        self.conv_margin = prototype.shape[0]

        assert peak_sign in ("both", "neg", "pos")
        self.nbefore = int(ms_before * recording.sampling_frequency / 1000)
        if peak_sign == "neg":
            assert prototype[self.nbefore] < 0, "Prototype should have a negative peak"
            peak_sign = "pos"
        elif peak_sign == "pos":
            assert prototype[self.nbefore] > 0, "Prototype should have a positive peak"

        self.peak_sign = peak_sign
        self.prototype = np.flip(prototype) / np.linalg.norm(prototype)

        contact_locations = recording.get_channel_locations()
        dist = np.linalg.norm(contact_locations[:, np.newaxis] - contact_locations[np.newaxis, :], axis=2)
        self.weights, self.z_factors = get_convolution_weights(dist, **weight_method)
        self.num_z_factors = len(self.z_factors)
        self.num_channels = recording.get_num_channels()
        self.num_templates = self.num_channels
        if peak_sign == "both":
            self.weights = np.hstack((self.weights, self.weights))
            self.weights[:, self.num_templates :, :] *= -1
            self.num_templates *= 2

        self.weights = self.weights.reshape(self.num_templates * self.num_z_factors, -1)
        self.weights = csr_matrix(self.weights)
        random_data = get_random_data_chunks(recording, return_in_uV=False, **random_chunk_kwargs)
        conv_random_data = self.get_convolved_traces(random_data)
        medians = np.median(conv_random_data, axis=1)
        self.medians = medians[:, None]
        noise_levels = np.median(np.abs(conv_random_data - self.medians), axis=1) / 0.6744897501960817
        self.abs_thresholds = noise_levels * detect_threshold
        self._dtype = np.dtype(base_peak_dtype + [("z", "float32")])

    def get_dtype(self):
        return self._dtype

    def get_trace_margin(self):
        return self.exclude_sweep_size + self.conv_margin

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):

        assert HAVE_NUMBA, "You need to install numba"
        conv_traces = self.get_convolved_traces(traces)
        # conv_traces -= self.medians
        conv_traces /= self.abs_thresholds[:, None]
        conv_traces = conv_traces[:, self.conv_margin : -self.conv_margin]
        traces_center = conv_traces[:, self.exclude_sweep_size : -self.exclude_sweep_size]

        traces_center = traces_center.reshape(self.num_z_factors, self.num_templates, traces_center.shape[1])
        conv_traces = conv_traces.reshape(self.num_z_factors, self.num_templates, conv_traces.shape[1])
        peak_mask = traces_center > 1

        peak_mask = _numba_detect_peak_matched_filtering(
            conv_traces,
            traces_center,
            peak_mask,
            self.exclude_sweep_size,
            self.abs_thresholds,
            self.peak_sign,
            self.neighbours_mask,
            self.num_channels,
        )

        # Find peaks and correct for time shift
        z_ind, peak_chan_ind, peak_sample_ind = np.nonzero(peak_mask)
        if self.peak_sign == "both":
            peak_chan_ind = peak_chan_ind % self.num_channels

        # If we want to estimate z
        # peak_chan_ind = peak_chan_ind % num_channels
        # z = np.zeros(len(peak_sample_ind), dtype=np.float32)
        # for count in range(len(peak_chan_ind)):
        #     channel = peak_chan_ind[count]
        #     peak = peak_sample_ind[count]
        #     data = traces[channel::num_channels, peak]
        #     z[count] = np.dot(data, z_factors)/data.sum()

        if peak_sample_ind.size == 0 or peak_chan_ind.size == 0:
            return (np.zeros(0, dtype=self._dtype),)

        peak_sample_ind += self.exclude_sweep_size + self.conv_margin + self.nbefore
        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]

        local_peaks = np.zeros(peak_sample_ind.size, dtype=self._dtype)
        local_peaks["sample_index"] = peak_sample_ind
        local_peaks["channel_index"] = peak_chan_ind
        local_peaks["amplitude"] = peak_amplitude
        local_peaks["segment_index"] = segment_index
        local_peaks["z"] = z_ind

        # return is always a tuple
        return (local_peaks,)

    def get_convolved_traces(self, traces):
        from scipy.signal import oaconvolve

        tmp = oaconvolve(self.prototype[None, :], traces.T, axes=1, mode="valid")
        scalar_products = self.weights.dot(tmp)
        return scalar_products


class DetectPeakLocallyExclusiveTorch(PeakDetectorWrapper):
    """Detect peaks using the "locally exclusive" method with pytorch."""

    name = "locally_exclusive_torch"
    engine = "torch"
    preferred_mp_context = "spawn"
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
        noise_levels=None,
        device=None,
        radius_um=50,
        return_tensor=False,
        random_chunk_kwargs={},
    ):
        if not HAVE_TORCH:
            raise ModuleNotFoundError('"by_channel_torch" needs torch which is not installed')
        args = DetectPeakByChannelTorch.check_params(
            recording,
            peak_sign=peak_sign,
            detect_threshold=detect_threshold,
            exclude_sweep_ms=exclude_sweep_ms,
            noise_levels=noise_levels,
            device=device,
            return_tensor=return_tensor,
            random_chunk_kwargs=random_chunk_kwargs,
        )

        channel_distance = get_channel_distances(recording)
        neighbour_indices_by_chan = []
        num_channels = recording.get_num_channels()
        for chan in range(num_channels):
            neighbour_indices_by_chan.append(np.nonzero(channel_distance[chan] <= radius_um)[0])
        max_neighbs = np.max([len(neigh) for neigh in neighbour_indices_by_chan])
        neighbours_idxs = num_channels * np.ones((num_channels, max_neighbs), dtype=int)
        for i, neigh in enumerate(neighbour_indices_by_chan):
            neighbours_idxs[i, : len(neigh)] = neigh
        return args + (neighbours_idxs,)

    @classmethod
    def get_method_margin(cls, *args):
        exclude_sweep_size = args[2]
        return exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, peak_sign, abs_thresholds, exclude_sweep_size, device, return_tensor, neighbor_idxs):
        sample_inds, chan_inds = _torch_detect_peaks(
            traces, peak_sign, abs_thresholds, exclude_sweep_size, neighbor_idxs, device
        )
        if not return_tensor and isinstance(sample_inds, torch.Tensor) and isinstance(chan_inds, torch.Tensor):
            sample_inds = np.array(sample_inds.cpu())
            chan_inds = np.array(chan_inds.cpu())
        return sample_inds, chan_inds






class DetectPeakLocallyExclusiveOpenCL(PeakDetectorWrapper):
    name = "locally_exclusive_cl"
    engine = "opencl"
    preferred_mp_context = None
    params_doc = (
        DetectPeakLocallyExclusive.params_doc
        + """
    opencl_context_kwargs: None or dict
        kwargs to create the opencl context
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
        # TODO refactor with other classes
        assert peak_sign in ("both", "neg", "pos")
        if noise_levels is None:
            noise_levels = get_noise_levels(recording, return_in_uV=False, **random_chunk_kwargs)
        abs_thresholds = noise_levels * detect_threshold
        exclude_sweep_size = int(exclude_sweep_ms * recording.get_sampling_frequency() / 1000.0)
        channel_distance = get_channel_distances(recording)
        neighbours_mask = channel_distance <= radius_um

        executor = OpenCLDetectPeakExecutor(abs_thresholds, exclude_sweep_size, neighbours_mask, peak_sign)

        return (executor,)

    @classmethod
    def get_method_margin(cls, *args):
        executor = args[0]
        return executor.exclude_sweep_size

    @classmethod
    def detect_peaks(cls, traces, executor):
        peak_sample_ind, peak_chan_ind = executor.detect_peak(traces)

        return peak_sample_ind, peak_chan_ind


class OpenCLDetectPeakExecutor:
    def __init__(self, abs_thresholds, exclude_sweep_size, neighbours_mask, peak_sign):

        self.chunk_size = None

        self.abs_thresholds = abs_thresholds.astype("float32")
        self.exclude_sweep_size = exclude_sweep_size
        self.neighbours_mask = neighbours_mask.astype("uint8")
        self.peak_sign = peak_sign
        self.ctx = None
        self.queue = None
        self.x = 0

    def create_buffers_and_compile(self, chunk_size):
        import pyopencl

        mf = pyopencl.mem_flags
        try:
            self.device = pyopencl.get_platforms()[0].get_devices()[0]
            self.ctx = pyopencl.Context(devices=[self.device])
        except Exception as e:
            print("error create context ", e)

        self.queue = pyopencl.CommandQueue(self.ctx)
        self.max_wg_size = self.ctx.devices[0].get_info(pyopencl.device_info.MAX_WORK_GROUP_SIZE)
        self.chunk_size = chunk_size

        self.neighbours_mask_cl = pyopencl.Buffer(
            self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.neighbours_mask
        )
        self.abs_thresholds_cl = pyopencl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.abs_thresholds)

        num_channels = self.neighbours_mask.shape[0]
        self.traces_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE, size=int(chunk_size * num_channels * 4))

        # TODO estimate smaller
        self.num_peaks = np.zeros(1, dtype="int32")
        self.num_peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.num_peaks)

        nb_max_spike_in_chunk = num_channels * chunk_size
        self.peaks = np.zeros(nb_max_spike_in_chunk, dtype=[("sample_index", "int32"), ("channel_index", "int32")])
        self.peaks_cl = pyopencl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=self.peaks)

        variables = dict(
            chunk_size=int(self.chunk_size),
            exclude_sweep_size=int(self.exclude_sweep_size),
            peak_sign={"pos": 1, "neg": -1}[self.peak_sign],
            num_channels=num_channels,
        )

        kernel_formated = processor_kernel % variables
        prg = pyopencl.Program(self.ctx, kernel_formated)
        self.opencl_prg = prg.build()  # options='-cl-mad-enable'
        self.kern_detect_peaks = getattr(self.opencl_prg, "detect_peaks")

        self.kern_detect_peaks.set_args(
            self.traces_cl, self.neighbours_mask_cl, self.abs_thresholds_cl, self.peaks_cl, self.num_peaks_cl
        )

        s = self.chunk_size - 2 * self.exclude_sweep_size
        self.global_size = (s,)
        self.local_size = None

    def detect_peak(self, traces):
        self.x += 1

        import pyopencl

        if self.chunk_size is None or self.chunk_size != traces.shape[0]:
            self.create_buffers_and_compile(traces.shape[0])
        event = pyopencl.enqueue_copy(self.queue, self.traces_cl, traces.astype("float32"))

        pyopencl.enqueue_nd_range_kernel(
            self.queue,
            self.kern_detect_peaks,
            self.global_size,
            self.local_size,
        )

        event = pyopencl.enqueue_copy(self.queue, self.traces_cl, traces.astype("float32"))
        event = pyopencl.enqueue_copy(self.queue, self.traces_cl, traces.astype("float32"))
        event = pyopencl.enqueue_copy(self.queue, self.num_peaks, self.num_peaks_cl)
        event = pyopencl.enqueue_copy(self.queue, self.peaks, self.peaks_cl)
        event.wait()

        n = self.num_peaks[0]
        peaks = self.peaks[:n]
        peak_sample_ind = peaks["sample_index"].astype("int64")
        peak_chan_ind = peaks["channel_index"].astype("int64")

        return peak_sample_ind, peak_chan_ind


processor_kernel = """
#define chunk_size %(chunk_size)d
#define exclude_sweep_size %(exclude_sweep_size)d
#define peak_sign %(peak_sign)d
#define num_channels %(num_channels)d


typedef struct st_peak{
    int sample_index;
    int channel_index;
} st_peak;


__kernel void detect_peaks(
                        //in
                        __global  float *traces,
                        __global  uchar *neighbours_mask,
                        __global  float *abs_thresholds,
                        //out
                        __global  st_peak *peaks,
                        volatile __global int *num_peaks
                ){
    int pos = get_global_id(0);

    if (pos == 0){
        *num_peaks = 0;
    }
    // this barrier OK if the first group is run first
    barrier(CLK_GLOBAL_MEM_FENCE);

    if (pos>=(chunk_size - (2 * exclude_sweep_size))){
        return;
    }


    float v;
    uchar peak;
    uchar is_neighbour;

    int index;

    int i_peak;


    for (int chan=0; chan<num_channels; chan++){

        //v = traces[(pos + exclude_sweep_size)*num_channels + chan];
        index = (pos + exclude_sweep_size) * num_channels + chan;
        v = traces[index];

        if(peak_sign==1){
            if (v>abs_thresholds[chan]){peak=1;}
            else {peak=0;}
        }
        else if(peak_sign==-1){
            if (v<-abs_thresholds[chan]){peak=1;}
            else {peak=0;}
        }

        if (peak == 1){
            for (int chan_neigh=0; chan_neigh<num_channels; chan_neigh++){

                is_neighbour = neighbours_mask[chan * num_channels + chan_neigh];
                if (is_neighbour == 0){continue;}
                //if (chan == chan_neigh){continue;}

                index = (pos + exclude_sweep_size) * num_channels + chan_neigh;
                if(peak_sign==1){
                    peak = peak && (v>=traces[index]);
                }
                else if(peak_sign==-1){
                    peak = peak && (v<=traces[index]);
                }

                if (peak==0){break;}

                if(peak_sign==1){
                    for (int i=1; i<=exclude_sweep_size; i++){
                        peak = peak && (v>traces[(pos + exclude_sweep_size - i)*num_channels + chan_neigh]) && (v>=traces[(pos + exclude_sweep_size + i)*num_channels + chan_neigh]);
                        if (peak==0){break;}
                    }
                }
                else if(peak_sign==-1){
                    for (int i=1; i<=exclude_sweep_size; i++){
                        peak = peak && (v<traces[(pos + exclude_sweep_size - i)*num_channels + chan_neigh]) && (v<=traces[(pos + exclude_sweep_size + i)*num_channels + chan_neigh]);
                        if (peak==0){break;}
                    }
                }

            }

        }

        if (peak==1){
            //append to
            i_peak = atomic_inc(num_peaks);
            // sample_index is LOCAL to fifo
            peaks[i_peak].sample_index = pos + exclude_sweep_size;
            peaks[i_peak].channel_index = chan;
        }
    }

}
"""


# TODO make a dict with name+engine entry later
_methods_list = [
    DetectPeakByChannel,
    DetectPeakLocallyExclusive,
    DetectPeakLocallyExclusiveOpenCL,
    DetectPeakByChannelTorch,
    DetectPeakLocallyExclusiveTorch,
    DetectPeakMatchedFiltering,
]
detect_peak_methods = {m.name: m for m in _methods_list}
method_doc = make_multi_method_doc(_methods_list)
detect_peaks.__doc__ = detect_peaks.__doc__.format(method_doc=method_doc, job_doc=_shared_job_kwargs_doc)
