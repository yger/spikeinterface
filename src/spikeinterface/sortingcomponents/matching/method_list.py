from __future__ import annotations

from .nearest import NearestTemplatesPeeler, NearestTemplatesSVDPeeler
from .tdc_peeler import TridesclousPeeler
from .circus import CircusOMPPeeler
from .wobble import WobbleMatch

matching_methods = {
    "nearest": NearestTemplatesPeeler,
<<<<<<< HEAD
    "nearest-svd" : NearestTemplatesSVDPeeler,
=======
    "nearest-svd": NearestTemplatesSVDPeeler,
>>>>>>> ae1a0d83f0ef3c883f61af1184320b0331684c7c
    "tdc-peeler": TridesclousPeeler,
    "circus-omp": CircusOMPPeeler,
    "wobble": WobbleMatch,
}

try:
    # Kilosort licence (GPL 3) is forcing us to make and use an external package
    from spikeinterface_kilosort_components.kilosort_matching import KiloSortMatching

    matching_methods["kilosort-matching"] = KiloSortMatching
except ImportError:
    pass
