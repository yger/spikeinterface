from __future__ import annotations
import numpy as np
from spikeinterface.sortingcomponents.clustering.peak_svd import extract_peaks_svd
from spikeinterface.sortingcomponents.peak_selection import select_peaks
from spikeinterface.sortingcomponents.tools import extract_waveform_at_max_channel

class SVDBasedClustering:
    """
    hdbscan clustering on peak_locations previously done by localize_peaks()
    """

    params_doc = ""
    _default_params = {
        "n_svd": None,
        "extract_peaks_svd_kwargs": {"ms_before": 0.5,
                                     "ms_after": 1.5,
                                     "radius_um": 100.0,
                                     "seed": None,
                                     "features_folder": None},
        "select_peaks_kwargs": {"method": "uniform",
                                "n_peaks": 10000,
                                "margin": (10, 10),
                                "seed": None},
    }

    @classmethod
    def compute_svd_features(cls, recording, peaks, params, job_kwargs=dict()):

        from sklearn.decomposition import TruncatedSVD

        ms_before = params["extract_peaks_svd_kwargs"].get("ms_before", 0.5)
        ms_after = params["extract_peaks_svd_kwargs"].get("ms_after", 1.5)
        fs = recording.get_sampling_frequency()
        nbefore = int(ms_before * fs / 1000.0)
        nafter = int(ms_after * fs / 1000.0)

        svd_model = TruncatedSVD(params["n_svd"], random_state=params["seed"])
        few_peaks = select_peaks(
            peaks,
            recording=recording,
            margin=(nbefore, nafter),
            **params["select_peaks_kwargs"])
        
        few_wfs = extract_waveform_at_max_channel(
            recording, few_peaks, ms_before=ms_before, ms_after=ms_after, **job_kwargs
        )
        
        wfs = few_wfs[:, :, 0]
        svd_model.fit(wfs)

        peaks_svd, sparse_mask, svd_model = extract_peaks_svd(
            recording,
            peaks,
            svd_model=svd_model,
            **params["extract_peaks_svd_kwargs"],
            **job_kwargs,
        )

        return peaks_svd, sparse_mask, svd_model

    @classmethod
    def merge_from_templates(cls, peaks, peak_labels, templates, sparse_mask, params):
        
        from spikeinterface.core import Templates
        from spikeinterface.sortingcomponents.clustering.merge_templates import merge_peak_labels_from_templates

        peak_labels, merge_template_array, merge_sparsity_mask, new_unit_ids = merge_peak_labels_from_templates(
            peaks,
            peak_labels,
            templates.unit_ids,
            templates.templates_array,
            sparse_mask,
            **params["merge_kwargs"],
        )

        fs = templates.sampling_frequency
        probe = templates.get_probe()
        channel_ids = templates.channel_ids

        templates = Templates(
            templates_array=merge_template_array,
            sampling_frequency=fs,
            nbefore=templates.nbefore,
            sparsity_mask=None,
            channel_ids=channel_ids,
            unit_ids=new_unit_ids,
            probe=probe,
            is_in_uV=False,
        )

        return peak_labels, templates, merge_sparsity_mask

    @classmethod
    def merge_from_features(cls, peaks, peak_labels, templates, sparse_mask, params):
        
        from spikeinterface.core import Templates
        from spikeinterface.sortingcomponents.clustering.merge_templates import merge_peak_labels_from_templates

        peak_labels, merge_template_array, merge_sparsity_mask, new_unit_ids = merge_peak_labels_from_templates(
            peaks,
            peak_labels,
            templates.unit_ids,
            templates.templates_array,
            sparse_mask,
            **params["merge_kwargs"],
        )

        fs = templates.sampling_frequency
        probe = templates.get_probe()
        channel_ids = templates.channel_ids

        templates = Templates(
            templates_array=merge_template_array,
            sampling_frequency=fs,
            nbefore=templates.nbefore,
            sparsity_mask=None,
            channel_ids=channel_ids,
            unit_ids=new_unit_ids,
            probe=probe,
            is_in_uV=False,
        )

        return peak_labels, templates, merge_sparsity_mask