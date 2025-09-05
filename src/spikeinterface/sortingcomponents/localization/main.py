from __future__ import annotations

import copy
import numpy as np
from .method_list import *

from spikeinterface.core.job_tools import (
    split_job_kwargs,
)

from spikeinterface.core.node_pipeline import (
    run_node_pipeline,
)

def localize_peaks(recording, peaks, method="center_of_mass", ms_before=0.5, ms_after=0.5, **kwargs) -> np.ndarray:
    """Localize peak (spike) in 2D or 3D depending the method.

    When a probe is 2D then:
       * X is axis 0 of the probe
       * Y is axis 1 of the probe
       * Z is orthogonal to the plane of the probe

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object.
    peaks : array
        Peaks array, as returned by detect_peaks() in "compact_numpy" way.
    ms_before : float
        The number of milliseconds to include before the peak of the spike
    ms_after : float
        The number of milliseconds to include after the peak of the spike

    {method_doc}

    {job_doc}

    Returns
    -------
    peak_locations: ndarray
        Array with estimated location for each spike.
        The dtype depends on the method. ("x", "y") or ("x", "y", "z", "alpha").
    """
    _, job_kwargs = split_job_kwargs(kwargs)
    peak_retriever = PeakRetriever(recording, peaks)
    pipeline_nodes = get_localization_pipeline_nodes(
        recording, peak_retriever, method=method, ms_before=ms_before, ms_after=ms_after, **kwargs
    )
    job_name = f"localize peaks using {method}"
    peak_locations = run_node_pipeline(recording, pipeline_nodes, job_kwargs, job_name=job_name, squeeze_output=True)

    return peak_locations