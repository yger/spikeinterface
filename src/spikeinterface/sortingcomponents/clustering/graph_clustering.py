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
        "ms_before" : 2,
        "ms_after" : 2,
        "motion": None,
        "seed": None,
        "clustering_method": "hdbscan",
        "clustering_kwargs" : dict(min_samples=1,
                                   min_cluster_size=50,
                                   n_jobs=-1,
                                   cluster_selection_method='leaf',
                                   cluster_selection_epsilon=0.1,
                                   allow_single_cluster=True),
        "peak_locations" : None,
        "graph_kwargs" : dict(normed_distances=True,
                              n_neighbors=50,
                              n_components=10,
                              bin_mode="channels",
                              sparse_mode="knn",
                              neighbors_radius_um=50,
                              apply_local_svd=True,
                              direction="y"),
        "extract_peaks_svd_kwargs" : dict(n_components=5)
    }

    @classmethod
    def main_function(cls, recording, peaks, params, job_kwargs=dict()):

        radius_um = params["radius_um"]
        # bin_um = params["bin_um"]
        motion = params["motion"]
        seed = params["seed"]
        ms_before = params["ms_before"]
        ms_after = params["ms_after"]
        peak_locations = params["peak_locations"]
        clustering_method = params["clustering_method"]
        clustering_kwargs = params["clustering_kwargs"]
        extract_peaks_svd_kwargs = params["extract_peaks_svd_kwargs"]
        graph_kwargs = params["graph_kwargs"]

        motion_aware = motion is not None

        if graph_kwargs["bin_mode"] == "channels":
            assert radius_um >= graph_kwargs["neighbors_radius_um"] * 2
        elif graph_kwargs["bin_mode"] == "vertical_bins":
            assert radius_um >= graph_kwargs["bin_um"] * 3

        peaks_svd, sparse_mask, svd_model = extract_peaks_svd(
            recording, peaks,
            radius_um=radius_um,
            ms_before=ms_before,
            ms_after=ms_after,
            motion_aware=motion_aware,
            motion=None,
            **extract_peaks_svd_kwargs,
            **job_kwargs
        )

        channel_locations = recording.get_channel_locations()
        channel_depth = channel_locations[:, 1]
        peak_depths = channel_depth[peaks["channel_index"]]


        # TODO : try to use real peak location
        
        # some method need a symetric matrix
        ensure_symetric = clustering_method in ("hdbscan", )

        distances = create_graph_from_peak_features(
            recording,
            peaks,
            peaks_svd,
            sparse_mask,
            peak_locations=None,
            # bin_um=bin_um,
            ensure_symetric=ensure_symetric,
            **graph_kwargs
        )

        #import scipy.sparse
        #scipy.sparse.save_npz("distances.npz", distances)

        # print(distances)
        # print(distances.shape)
        # print("sparsity: ", distances.indices.size / (distances.shape[0]**2))        


        print("clustering_method", clustering_method)

        if clustering_method == "networkx-louvain":
            # using networkx : very slow (possible backend with cude  backend="cugraph",)
            import networkx as nx

            distances_bool = distances.copy()
            distances_bool.data[:] = 1
            G = nx.Graph(distances_bool)
            communities = nx.community.louvain_communities(G, seed=seed)
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
            classifier = Louvain()
            distances_bool = distances.copy()
            distances_bool.data[:] = 1
            peak_labels = classifier.fit_predict(distances_bool)
            _remove_small_cluster(peak_labels, min_size=1)

        elif clustering_method == "sknetwork-leiden":
            from sknetwork.clustering import Leiden
            classifier = Leiden()
            distances_bool = distances.copy()
            distances_bool.data[:] = 1
            peak_labels = classifier.fit_predict(distances_bool)
            _remove_small_cluster(peak_labels, min_size=1)

        elif clustering_method == "leidenalg":
            import leidenalg
            import igraph
            adjacency = distances.copy()
            adjacency.data = 1. - adjacency.data  
            graph = igraph.Graph.Weighted_Adjacency(adjacency.tocoo(), mode='directed',)
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
            from scipy.sparse.csgraph import connected_components
            n_components, labels = connected_components(csgraph=distances, 
                                                        directed=False, 
                                                        return_labels=True)
            peak_labels = -1*np.ones(len(peaks), dtype=int)
            n_clusters = 0

            for component in range(n_components):
                connected_nodes = np.flatnonzero(labels == component)
                if len(connected_nodes) > 1:
                    sub_distances = distances[connected_nodes].tocsc()[:, connected_nodes]
                    clusterer = HDBSCAN(metric='precomputed',
                                        **clustering_kwargs)
                    clusterer.fit(sub_distances)
                    valid_clusters = np.flatnonzero(clusterer.labels_ > -1)
                    if valid_clusters.size > 0:
                        peak_labels[connected_nodes[valid_clusters]] = clusterer.labels_[valid_clusters] + n_clusters
                        n_clusters = np.max(clusterer.labels_[valid_clusters]) + 1
        else:
            raise ValueError("GraphClustering : wrong clustering_method")

        labels_set = np.unique(peak_labels)
        labels_set = labels_set[labels_set >= 0]        

        fs = recording.get_sampling_frequency()
        nbefore = int(ms_before * fs / 1000.0)
        nafter = int(ms_after * fs / 1000.0)
        num_channels = recording.get_num_channels()
        templates_array = np.zeros((len(labels_set), nbefore+nafter, num_channels), dtype=np.float32)
        for unit_ind, label in enumerate(labels_set):
            mask = peak_labels == label
            local_peaks = peaks[mask]
            local_svd = peaks_svd[mask]
            peak_channels, b = np.unique(local_peaks['channel_index'], return_counts=True)
            best_channel = peak_channels[np.argmax(b)]
            sub_mask = local_peaks['channel_index'] == best_channel
            for count, i in enumerate(np.flatnonzero(sparse_mask[best_channel])):
                data = svd_model.inverse_transform(local_svd[sub_mask, :, count])
                templates_array[unit_ind, :, i] = np.median(data, 0)

        unit_ids = np.arange(len(labels_set))

        from spikeinterface.core.template import Templates
        templates = Templates(
            templates_array=templates_array,
            sampling_frequency=fs,
            nbefore=nbefore,
            sparsity_mask=None,
            channel_ids=recording.channel_ids,
            unit_ids=unit_ids,
            probe=recording.get_probe(),
            is_scaled=False,
        )

        return labels_set, peak_labels, templates



def _remove_small_cluster(peak_labels, min_size=1):
    for k in np.unique(peak_labels):
        inds = np.flatnonzero(peak_labels == k)
        if inds.size <= min_size:
            peak_labels[inds] = -1            

