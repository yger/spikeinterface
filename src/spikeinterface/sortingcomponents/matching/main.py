from __future__ import annotations

import numpy as np

from .method_list import *

# from spikeinterface.core.job_tools import ChunkRecordingExecutor, fix_job_kwargs
# from spikeinterface.core import get_chunk_with_margin

from spikeinterface.core.job_tools import (
    split_job_kwargs,
    _shared_job_kwargs_doc
)

from spikeinterface.core.job_tools import fix_job_kwargs
from spikeinterface.core.node_pipeline import run_node_pipeline


def find_spikes_from_templates(
    recording,
    templates,
    method="naive",
    extra_outputs=False,
    gather_mode="memory",
    gather_kwargs=None,
    **kwargs,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Find spike from a recording from given templates.

    Parameters
    ----------
    recording : RecordingExtractor
        The recording extractor object
    templates : Templates
        The Templates that should be look for in the data
    method : "naive" | "tridesclous" | "circus" | "circus-omp" | "wobble", default: "naive"
        Which method to use for template matching
    method_kwargs : dict, optional
        Keyword arguments for the chosen method
    extra_outputs : bool
        If True then a dict is also returned is also returned
    gather_mode : "memory" | "npy", default: "memory"
        If "memory" then the output is gathered in memory, if "npy" then the output is gathered on disk
    gather_kwargs : dict, optional
        The kwargs for the gather method

    {method_doc}

    {job_doc}

    Returns
    -------
    spikes : ndarray
        Spikes found from templates.
    outputs:
        Optionaly returns for debug purpose.
    """
    from spikeinterface.sortingcomponents.matching.method_list import matching_methods

    assert method in matching_methods, f"Method {method} is not supported. Choose from {matching_methods}"

    method_kwargs, job_kwargs = split_job_kwargs(kwargs)
    job_kwargs = fix_job_kwargs(job_kwargs)

    method_class = matching_methods[method]

    if len(templates.unit_ids) == 0:
        return np.zeros(0, dtype=node0.get_dtype())
    else:
        method_kwargs.update({"templates" : templates})

    node0 = method_class(recording, **method_kwargs)
    nodes = [node0]

    gather_kwargs = gather_kwargs or {}
    names = ["spikes"]

    spikes = run_node_pipeline(
        recording,
        nodes,
        job_kwargs,
        job_name=f"find spikes ({method})",
        gather_mode=gather_mode,
        squeeze_output=True,
        names=names,
        **gather_kwargs,
    )

    if extra_outputs:
        outputs = node0.get_extra_outputs()

    node0.clean()

    if extra_outputs:
        return spikes, outputs
    else:
        return spikes


method_doc = make_multi_method_doc(list(matching_methods.values()))
find_spikes_from_templates.__doc__ = find_spikes_from_templates.__doc__.format(method_doc=method_doc, job_doc=_shared_job_kwargs_doc)