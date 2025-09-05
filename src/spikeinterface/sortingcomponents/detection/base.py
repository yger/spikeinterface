from spikeinterface.core.node_pipeline import (
    PeakDetector,
    base_peak_dtype,
)
import numpy as np

class PeakDetectorWrapper(PeakDetector):
    # transitory class to maintain instance based and class method based
    # TODO later when in main: refactor in every old detector class:
    #    * check_params
    #    * get_method_margin
    #  and move the logic in the init
    #  but keep the class method "detect_peaks()" because it is convinient in template matching
    def __init__(self, recording, **params):
        PeakDetector.__init__(self, recording, return_output=True)

        self.params = params
        self.args = self.check_params(recording, **params)

    def get_trace_margin(self):
        return self.get_method_margin(*self.args)

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        peak_sample_ind, peak_chan_ind = self.detect_peaks(traces, *self.args)
        if peak_sample_ind.size == 0 or peak_chan_ind.size == 0:
            return (np.zeros(0, dtype=base_peak_dtype),)

        peak_amplitude = traces[peak_sample_ind, peak_chan_ind]
        local_peaks = np.zeros(peak_sample_ind.size, dtype=base_peak_dtype)
        local_peaks["sample_index"] = peak_sample_ind
        local_peaks["channel_index"] = peak_chan_ind
        local_peaks["amplitude"] = peak_amplitude
        local_peaks["segment_index"] = segment_index

        # return is always a tuple
        return (local_peaks,)