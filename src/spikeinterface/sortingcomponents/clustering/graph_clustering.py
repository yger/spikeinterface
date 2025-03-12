from pathlib import Path

import numpy as np


from spikeinterface.sortingcomponents.clustering.peak_svd import extract_peaks_svd
from spikeinterface.sortingcomponents.clustering.graph_tools import create_graph_from_peak_features


class GraphClustering:
    """
    Simple clustering by constructing a global sparse graph using local slinding bins along the probe.

    The edge of the graph is constructed using local distance bewteen svd on waveforms.

    Then a classic algorithm like louvain or hdbscan is used.
    """

    _default_params = {
        "radius_um": 100.,
        "bin_um": 30.,
        "motion": None,
        "seed": None,
        "n_neighbors": 100,
        "clustering_method": "hdbscan",
        "clustering_kwargs" : dict(min_samples=1, n_jobs=-1, min_cluster_size=200),
        "peak_locations" : None
    }

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):

        radius_um = params["radius_um"]
        bin_um = params["bin_um"]
        motion = params["motion"]
        seed = params["seed"]
        n_neighbors = params["n_neighbors"]
        peak_locations = params["peak_locations"]
        clustering_method = params["clustering_method"]
        clustering_kwargs = params["clustering_kwargs"]

        motion_aware = motion is not None

        assert radius_um >= bin_um * 3

        peaks_svd, sparse_mask, _ = extract_peaks_svd(
            recording, peaks,
            radius_um=radius_um,
            motion_aware=motion_aware,
            motion=None,
        )

        if peak_locations is None:
            channel_locations = recording.get_channel_locations()
            channel_depth = channel_locations[:, 1]
            peak_depths = channel_depth[peaks["channel_index"]]
        else:
            peak_depths = peak_locations[:, 1]

        distances = create_graph_from_peak_features(
            recording,
            peaks,
            peaks_svd,
            sparse_mask,
            peak_locations=None,
            bin_um=bin_um,
            dim=1,
            #mode="radius",
            mode="knn",
            direction="y",
            n_neighbors=n_neighbors,
        )

        # import scipy.sparse
        # scipy.sparse.save_npz('test', distances)
        
        print("clustering_method", clustering_method)

        if clustering_method == "networkx-louvain":
            # using networkx : very slow (possible backend with cude  backend="cugraph",)
            import networkx as nx
            distances.data[:] = 1
            G = nx.Graph(distances)
            communities = nx.community.louvain_communities(G, seed=seed, **clustering_kwargs)
            peak_labels = np.zeros(peaks.size, dtype=int)
            peak_labels[:] = -1
            k = 0
            for community in communities:
                if len(community) == 1:
                    continue
                peak_labels[list(community)] = k
                k += 1
        
        elif clustering_method == "sknetwork-louvain":
            from sknetwork.clustering import Louvain
            distances.data[:] = 1
            classifier = Louvain(**clustering_kwargs)
            peak_labels = classifier.fit_predict(distances)
            _remove_small_cluster(peak_labels, min_size=1)

        elif clustering_method == "sknetwork-leiden":
            from sknetwork.clustering import Leiden
            distances.data[:] = 1
            classifier = Leiden(**clustering_kwargs)
            peak_labels = classifier.fit_predict(distances)
            _remove_small_cluster(peak_labels, min_size=1)

        elif clustering_method == "leidenalg":
            import leidenalg
            import igraph
            graph = igraph.Graph.Weighted_Adjacency(distances.tocoo(), mode='directed',)
            clusters = leidenalg.find_partition(graph, leidenalg.ModularityVertexPartition)
            peak_labels = np.array(clusters.membership)
            _remove_small_cluster(peak_labels, min_size=1)

        elif clustering_method == "igraph-label-propagation":
            import igraph
            graph = igraph.Graph.Weighted_Adjacency(distances.tocoo(), mode='directed',)
            clusters = graph.community_label_propagation()
            peak_labels = np.array(clusters.membership)
            _remove_small_cluster(peak_labels, min_size=1)
        elif clustering_method == "hdbscan":
            from sklearn.cluster import HDBSCAN
            symmetric = distances.maximum(distances.T)
            clusterer = HDBSCAN(metric='precomputed', 
                                metric_params={'max_distance' : np.inf},
                                **clustering_kwargs)
            
            clusterer.fit(symmetric)
            peak_labels = clusterer.labels_ 
        else:
            raise ValueError("GraphClustering : wrong clustering_method")

        labels_set = np.unique(peak_labels)
        labels_set = labels_set[labels_set >= 0]        
        return labels_set, peak_labels



def _remove_small_cluster(peak_labels, min_size=1):
    for k in np.unique(peak_labels):
        inds = np.flatnonzero(peak_labels == k)
        if inds.size <= min_size:
            peak_labels[inds] = -1            

