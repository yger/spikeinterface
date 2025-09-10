from __future__ import annotations
import numpy as np
import importlib.util

hdbscan_spec = importlib.util.find_spec("hdbscan")
if hdbscan_spec is not None:
    HAVE_HDBSCAN = True
else:
    HAVE_HDBSCAN = False

class PositionClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """

    name = "position"
    params_doc = ""
    _default_params = {
        "peak_locations": None,
        "peak_localization_kwargs": {"method": "center_of_mass"},
        "hdbscan_kwargs": {"min_cluster_size": 20, "allow_single_cluster": True, "core_dist_n_jobs": -1},
        "tmp_folder": None,
    }

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):
        assert HAVE_HDBSCAN, "position clustering need hdbscan to be installed"
        import hdbscan
        d = params

        if d["peak_locations"] is None:
            from spikeinterface.sortingcomponents.peak_localization import localize_peaks
            peak_locations = localize_peaks(recording, peaks, **d["peak_localization_kwargs"], **job_kwargs)
        else:
            peak_locations = d["peak_locations"]

        tmp_folder = d["tmp_folder"]
        if tmp_folder is not None:
            tmp_folder.mkdir(exist_ok=True)

        location_keys = ["x", "y"]
        locations = np.stack([peak_locations[k] for k in location_keys], axis=1)
        to_cluster_from = locations

        clustering = hdbscan.hdbscan(to_cluster_from, **d["hdbscan_kwargs"])
        peak_labels = clustering[0]

        labels = np.unique(peak_labels)
        labels = labels[labels >= 0]

        return labels, peak_labels
