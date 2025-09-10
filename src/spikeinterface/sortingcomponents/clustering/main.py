from __future__ import annotations

from .method_list import *

from spikeinterface.core.job_tools import fix_job_kwargs, _shared_job_kwargs_doc

from spikeinterface.core.job_tools import (
    split_job_kwargs,
    _shared_job_kwargs_doc,
    fix_job_kwargs
)

from ..tools import make_multi_method_doc

def find_clusters_from_peaks(recording, peaks, method="stupid", extra_outputs=False, **kwargs):
    """
    Find cluster from peaks.


    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object
    peaks : numpy.array
        The peak vector
    method : str
        Which method to use ("stupid" | "XXXX")
    method_kwargs : dict, default: dict()
        Keyword arguments for the chosen method
    extra_outputs : bool, default: False
        If True then debug is also return
    {}

    Returns
    -------
    labels_set: ndarray of int
        possible clusters list
    peak_labels: array of int
        peak_labels.shape[0] == peaks.shape[0]
    """

    from spikeinterface.sortingcomponents.clustering.method_list import clustering_methods

    assert (
        method in clustering_methods
    ), f"Method for clustering do not exists, should be in {list(clustering_methods.keys())}"

    method_kwargs, job_kwargs = split_job_kwargs(kwargs)
    job_kwargs = fix_job_kwargs(job_kwargs)

    method_class = clustering_methods[method]
    params = method_class._default_params.copy()
    params.update(**method_kwargs)

    outputs = method_class.main_function(recording, peaks, params, job_kwargs=job_kwargs)

    if extra_outputs:
        return outputs
    else:
        if len(outputs) > 2:
            outputs = outputs[:2]
        labels_set, peak_labels = outputs
        return labels_set, peak_labels


method_doc = make_multi_method_doc(list(clustering_methods.values()))
find_clusters_from_peaks.__doc__ = find_clusters_from_peaks.__doc__.format(method_doc=method_doc, job_doc=_shared_job_kwargs_doc)
