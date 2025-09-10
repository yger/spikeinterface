from __future__ import annotations

from .center_of_mass import LocalizeCenterOfMass
from .monopolar import LocalizeMonopolarTriangulation
from .grid import LocalizeGridConvolution

localization_methods = {
    "center_of_mass": LocalizeCenterOfMass,
    "monopolar_triangulation": LocalizeMonopolarTriangulation,
    "grid_convolution": LocalizeGridConvolution,
}