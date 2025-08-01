import pytest
import numpy as np

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.sortingcomponents.peak_localization import localize_peaks
from spikeinterface.sortingcomponents.clustering import find_cluster_from_peaks
from spikeinterface.sortingcomponents.clustering.method_list import clustering_methods
from spikeinterface.sortingcomponents.clustering.peak_svd import extract_peaks_svd
from spikeinterface.sortingcomponents.clustering.graph_tools import create_graph_from_peak_features
from spikeinterface.sortingcomponents.clustering.tools import get_templates_from_peaks_and_svd

from spikeinterface.core import get_noise_levels

from spikeinterface.sortingcomponents.tests.common import make_dataset

import time

import importlib


def job_kwargs():
    return dict(n_jobs=1, chunk_size=10000, progress_bar=True, mp_context="spawn")


@pytest.fixture(name="job_kwargs", scope="module")
def job_kwargs_fixture():
    return job_kwargs()


@pytest.fixture(name="recording", scope="module")
def recording():
    rec, sorting = make_dataset()
    print(rec)
    return rec


def run_peaks(recording, job_kwargs):
    noise_levels = get_noise_levels(recording, return_in_uV=False)
    return detect_peaks(
        recording,
        method="locally_exclusive",
        peak_sign="neg",
        detect_threshold=5,
        exclude_sweep_ms=0.1,
        noise_levels=noise_levels,
        **job_kwargs,
    )


@pytest.fixture(name="peaks", scope="module")
def peaks_fixture(recording, job_kwargs):
    return run_peaks(recording, job_kwargs)


def run_peak_locations(recording, peaks, job_kwargs):
    return localize_peaks(recording, peaks, method="center_of_mass", **job_kwargs)


@pytest.fixture(name="peak_locations", scope="module")
def peak_locations_fixture(recording, peaks, job_kwargs):
    return run_peak_locations(recording, peaks, job_kwargs)


clustering_method_keys = list(clustering_methods.keys())
have_isosplit6 = importlib.util.find_spec("isosplit6") is not None

if "tdc-clustering" in clustering_method_keys and not have_isosplit6:
    # skip tdc-clustering if not isosplit6
    clustering_method_keys.remove("tdc-clustering")


@pytest.mark.parametrize("clustering_method", clustering_method_keys)
def test_find_cluster_from_peaks(clustering_method, recording, peaks, peak_locations):
    method_kwargs = {}
    if clustering_method in ("position", "position_and_pca"):
        method_kwargs["peak_locations"] = peak_locations
    if clustering_method in ("sliding_hdbscan", "position_and_pca"):
        method_kwargs["waveform_mode"] = "shared_memory"

    t0 = time.perf_counter()
    labels, peak_labels = find_cluster_from_peaks(
        recording, peaks, method=clustering_method, method_kwargs=method_kwargs
    )
    t1 = time.perf_counter()
    print(clustering_method, "found", len(labels), "clusters in ", t1 - t0)


def test_extract_peaks_svd(recording, peaks, job_kwargs):
    peaks_svd, sparse_mask, svd_model = extract_peaks_svd(recording, peaks, n_components=5, **job_kwargs)
    assert peaks_svd.shape[0] == peaks.shape[0]
    assert peaks_svd.shape[1] == 5
    assert peaks_svd.shape[2] == np.max(np.sum(sparse_mask, axis=1))


def test_create_graph_from_peak_features(recording, peaks, job_kwargs):
    peaks_svd, sparse_mask, svd_model = extract_peaks_svd(recording, peaks, n_components=5, **job_kwargs)

    distances = create_graph_from_peak_features(
        recording,
        peaks,
        peaks_svd,
        sparse_mask,
    )
    print(distances.shape)


def test_templates_from_svd(recording, peaks, job_kwargs):
    peaks_svd, sparse_mask, svd_model = extract_peaks_svd(
        recording, peaks, n_components=1, ms_before=1, ms_after=1, **job_kwargs
    )
    templates = get_templates_from_peaks_and_svd(
        recording,
        peaks,
        peaks["channel_index"],
        ms_before=1,
        ms_after=1,
        svd_features=peaks_svd,
        sparsity_mask=sparse_mask,
        svd_model=svd_model,
    )


if __name__ == "__main__":
    job_kwargs = dict(n_jobs=1, chunk_size=10000, progress_bar=True)
    recording, sorting = make_dataset()
    peaks = run_peaks(recording, job_kwargs)
    peak_locations = run_peak_locations(recording, peaks, job_kwargs)
    # method = "position_and_pca"
    method = "circus-clustering"
    # method = "tdc-clustering"
    # method = "random_projections"
    # method = "graph-clustering"

    test_find_cluster_from_peaks(method, recording, peaks, peak_locations)

    # test_extract_peaks_svd(recording, peaks, job_kwargs)
    # test_create_graph_from_peak_features(recording, peaks, job_kwargs)
