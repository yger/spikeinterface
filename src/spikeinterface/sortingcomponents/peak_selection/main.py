"""Sorting components: peak selection"""

from __future__ import annotations
import numpy as np

from .method_list import *
from ..tools import make_multi_method_doc

from spikeinterface.core.job_tools import (
    split_job_kwargs,
    _shared_job_kwargs_doc,
    fix_job_kwargs
)

def select_peaks(
    peaks, 
    method="uniform", 
    recording=None, 
    seed=None, 
    return_indices=False, 
    margin=None,
    **kwargs
):
    """
    Method to select a subset of peaks from a set of peaks.
    Usually use for reducing computational foorptint of downstream methods.
    Parameters
    ----------
    peaks: the peaks that have been found
    method: str
        The selection method to use. See `selection_methods` for available methods.
    seed: int
        The seed for random generations
    return_indices: bool
        If True, return the indices of selection such that selected_peaks = peaks[selected_indices]
    margin : Margin in timesteps. default: None. Otherwise should be a tuple (nbefore, nafter)
        preventing peaks to be selected at the borders of the segments. A recording should be provided to get the duration
        of the segments

    {method_doc}

    {job_doc}  

    Returns
    -------
    selected_peaks: array
        Selected peaks.
    selected_indices: array
        indices of peak selection such that selected_peaks = peaks[selected_indices].  Only returned when
        return_indices is True.
    """

    assert (
        method in selection_methods
    ), f"Method {method} is not supported. Choose from {selection_methods.keys()}"

    if margin is not None:
        assert recording is not None, "recording should be provided if margin is not None"

    selected_indices = _select_peak_indices(peaks, method=method, recording=recording, seed=seed, **kwargs)
    selected_peaks = peaks[selected_indices]
    num_segments = len(np.unique(selected_peaks["segment_index"]))

    if margin is not None:
        to_keep = np.zeros(len(selected_peaks), dtype=bool)
        for segment_index in range(num_segments):
            num_samples_in_segment = recording.get_num_samples(segment_index)
            i0, i1 = np.searchsorted(selected_peaks["segment_index"], [segment_index, segment_index + 1])
            while selected_peaks["sample_index"][i0] <= margin[0]:
                i0 += 1
            while selected_peaks["sample_index"][i1 - 1] >= (num_samples_in_segment - margin[1]):
                i1 -= 1
            to_keep[i0:i1] = True
        selected_indices = selected_indices[to_keep]
        selected_peaks = peaks[selected_indices]

    if return_indices:
        return selected_peaks, selected_indices
    else:
        return selected_peaks

def _select_peak_indices(peaks, method, recording, **kwargs):
    
    method_kwargs, job_kwargs = split_job_kwargs(kwargs)
    job_kwargs = fix_job_kwargs(job_kwargs)

    method_class = selection_methods[method]

    if method_class.need_noise_levels:
        assert recording is not None, f"recording should be provided for the method {method}"
        from spikeinterface.core.recording_tools import get_noise_levels
        random_chunk_kwargs = method_kwargs.pop("random_chunk_kwargs", {})
        if "noise_levels" not in method_kwargs:
            method_kwargs["noise_levels"] = get_noise_levels(
                recording, return_in_uV=False, **random_chunk_kwargs, **job_kwargs
            )

    selector = method_class(**method_kwargs)

    selected_indices = selector.compute(peaks)

    selected_indices = np.concatenate(selected_indices)
    selected_indices = selected_indices[
        np.lexsort((peaks[selected_indices]["sample_index"], peaks[selected_indices]["segment_index"]))
    ]
    return selected_indices


method_doc = make_multi_method_doc(list(selection_methods.values()))
select_peaks.__doc__ = select_peaks.__doc__.format(method_doc=method_doc, job_doc=_shared_job_kwargs_doc)