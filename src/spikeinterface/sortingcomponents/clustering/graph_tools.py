
import numpy as np

from tqdm.auto import tqdm

from spikeinterface.sortingcomponents.clustering.tools import aggregate_sparse_features



def create_graph_from_peak_features(
    recording,
    peaks,
    peak_features,
    sparse_mask,
    peak_locations=None,
    bin_um=20.,
    mode="knn",
    apply_local_svd=True,
    n_components=0.8,
    normed_distances=True,
    direction="y",
    n_neighbors=50,
    enforce_diagonal_to_zero=True,
    progress_bar=True,
):
    """
    Create a sparse garph of peaks distances.
    This done using a binarization along the depth axis.
    Each peaks can connect to peaks of the same bin and neighbour bins.

    The distances are locally computed on a local sparse set of channels that depend on thev detph.
    So the original features sparsity must be big enougth to cover local channel (actual bin+neighbour).

    2 possible modes:
      * "full" : compute the distances from all peaks in a bin to all peaks in the same bin + neighbour
      * "knn" : keep the k neareast neighbour for each peaks in bin + neighbour
      * "radius" : keep the spikes in bin that are within a given radius in um
    
    Important, peak_locations can be:
      * the peak location from the channel (fast)
      * the estimated peak location
      * the corrected peak location if the peak_features is computed with motion_awre in mind

    Note : the binarization works for linear probe only. This need to be extended to 2d grid binarization for planar mea.
    """

    import scipy.sparse
    from scipy.spatial.distance import cdist
    if mode in ["knn"]:
        from sklearn.neighbors import NearestNeighbors
    if apply_local_svd:
        from sklearn.decomposition import TruncatedSVD    

    dim = "xyz".index(direction)
    channel_locations = recording.get_channel_locations()
    channel_depth = channel_locations[:, dim]

    if peak_locations is None:
        # we use the max channel location instead
        peak_depths = channel_depth[peaks["channel_index"]]
    else:
        peak_depths = peak_locations[direction]

    # todo center this bons like in peak localization
    loc0 = min(channel_depth)
    loc1 = max(channel_depth)
    eps = bin_um/10.
    bins = np.arange(loc0-bin_um/2, loc1+bin_um/2+eps, bin_um)

    bins[0] = -np.inf
    bins[-1] = np.inf

    loop = range(bins.size-1)
    if progress_bar:
        loop = tqdm(loop, desc="Construct distance graph")

    local_graphs = []
    row_indices = []
    for b in loop:
        
        # limits for peaks
        l0, l1 = bins[b], bins[b+1]

        # limits for features
        b0, b1 = l0 - bin_um, l1 + bin_um
        local_chans = np.flatnonzero((channel_locations[:, dim] >= (b0 )) & (channel_locations[:, dim] <= (b1)))

        mask = (peak_depths >= b0) & (peak_depths < b1)
        peak_indices = np.flatnonzero(mask)

        local_depths = peak_depths[peak_indices]

        target_mask = (local_depths >= l0) & (local_depths < l1)
        
        local_feats, dont_have_channels = aggregate_sparse_features(peaks, peak_indices,
                                                                 peak_features, sparse_mask, local_chans)

        target_mask *= ~dont_have_channels
        #if np.sum(no_channels_target) > 0:
        #    print("dont_have_channels", np.sum(no_channels_target), "for n=", peak_indices.size, "bin", b0, b1)
        
        target_local_inds = np.flatnonzero(target_mask)
        target_indices = peak_indices[target_local_inds]

        if target_indices.size == 0:
            continue

        flatten_feat = local_feats.reshape(local_feats.shape[0], -1)
        row_indices.append(target_indices)

        if apply_local_svd:
            if isinstance(n_components, int):
                n_components = min(n_components, flatten_feat.shape[1])
                tsvd = TruncatedSVD(n_components)
                flatten_feat = tsvd.fit_transform(flatten_feat)
            elif isinstance(n_components, float):
                assert 0 < n_components < 1, "n_components should be in ]0, 1["       
                tsvd = TruncatedSVD(flatten_feat.shape[1])
                flatten_feat = tsvd.fit_transform(flatten_feat)
                thr = np.percentile(tsvd.explained_variance_ratio_, 100*n_components)
                indices = tsvd.explained_variance_ratio_ > thr
                flatten_feat = flatten_feat[:, indices]
        
        if mode == "full":
            local_dists = cdist(flatten_feat[target_mask], flatten_feat)

            if normed_distances:
                norm = np.linalg.norm(flatten_feat, axis=1)
                local_dists /= (norm[target_mask, None] + norm[None, :])

            data = local_dists.flatten().astype("float32")
            indptr = np.arange(0, local_dists.size + 1, local_dists.shape[1])
            indices = np.concatenate([peak_indices] * target_indices.size )
            local_graph = scipy.sparse.csr_matrix((data, indices, indptr), shape=(target_indices.size, peaks.size), dtype=np.float32)
            local_graphs.append(local_graph)
        elif mode == "knn":
            nn = min(n_neighbors, len(flatten_feat))
            if nn == 0 or flatten_feat.shape[1] == 0:
                local_graph = scipy.sparse.csr_matrix((target_indices.size, peaks.size), dtype=np.float32)
            else:
                nn_tree = NearestNeighbors(n_neighbors=nn, n_jobs=1, metric='l1')
                nn_tree.fit(flatten_feat)
                local_sparse_dist  = nn_tree.kneighbors_graph(flatten_feat[target_mask], mode='distance')
                data = local_sparse_dist.data.astype("float32")
                indptr = local_sparse_dist.indptr

                if normed_distances:    
                    for i in range(local_sparse_dist.shape[0]):
                        a, b = indptr[i], indptr[i+1]
                        src = flatten_feat[target_mask][i]
                        tgt = flatten_feat[local_sparse_dist.indices[a:b]]
                        norm = (np.linalg.norm(src, 1) + np.linalg.norm(tgt, 1, axis=1))
                        data[a:b] /= norm
                
                indices = peak_indices[local_sparse_dist.indices]
                local_graph = scipy.sparse.csr_matrix((data, indices, indptr), shape=(target_indices.size, peaks.size), dtype=np.float32)
            local_graphs.append(local_graph)

        else:
            raise ValueError("create_graph_from_peak_features() wrong mode")
    
    # stack all local distances in a big sparse one
    if len(local_graphs) > 0:
        distances = scipy.sparse.vstack(local_graphs)
        row_indices = np.concatenate(row_indices)
        row_order = np.argsort(row_indices)
        distances = distances[row_order]
        # t1 = time.perf_counter()
        # print("row_order", t1 - t0)

        if mode == "knn":
            # t0 = time.perf_counter()
            distances = scipy.sparse.csr_matrix(distances)
            # t1 = time.perf_counter()
            # print("final csr", t1 - t0)

        if enforce_diagonal_to_zero:
            ind0, ind1 = distances.tocoo().coords
            distances.data[ind0 == ind1] = 0.

    else:
        distances = scipy.sparse.csr_matrix((peaks.size, peaks.size), dtype="float32")

    return distances
