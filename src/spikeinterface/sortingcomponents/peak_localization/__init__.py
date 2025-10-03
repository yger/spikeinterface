# the method list is imported on the fly by find_spikes_from_templates otherwise it is not
# possible to inject external methods (like ks) dynamically
# from .method_list import matching_methods

from .method_list import peak_localization_methods
<<<<<<< HEAD
from .main import localize_peaks
=======
from .main import localize_peaks, get_localization_pipeline_nodes
>>>>>>> ae1a0d83f0ef3c883f61af1184320b0331684c7c
