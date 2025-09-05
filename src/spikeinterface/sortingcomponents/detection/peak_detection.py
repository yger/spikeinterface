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
