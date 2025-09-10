from __future__ import annotations

from .dummy import DummyClustering
from .position import PositionClustering

# from .sliding_nn import SlidingNNClustering
from .position_and_pca import PositionAndPCAClustering
from .random_projections import RandomProjectionClustering
from .iterative_hdbscan import IterHDBSCANClustering
from .iterative_isosplit import IterISOSPLITClustering
from .graph_clustering import GraphClustering

clustering_methods = {
    "dummy": DummyClustering,
    "position": PositionClustering,
    #"position_and_pca": PositionAndPCAClustering,
    #"random_projections": RandomProjectionClustering,
    #"iterative-hdbscan": IterHDBSCANClustering,
    #"iterative-isosplit": IterISOSPLITClustering,
    #"graph-clustering": GraphClustering,
}


#try:
#    # Kilosort licence (GPL 3) is forcing us to make and use an external package
#    from spikeinterface_kilosort_components.kilosort_clustering import KiloSortClustering
#
#    clustering_methods["kilosort-clustering"] = KiloSortClustering
#except ImportError:
#    pass
