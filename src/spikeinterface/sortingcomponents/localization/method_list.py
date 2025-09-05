from __future__ import annotations

from .center_of_mass import LocalizeCenterOfMass
from .monopolar import LocalizeMonopolarTriangulation
from .grid import LocalizeGridConvolution

matching_methods = {
    "center_of_mass": LocalizeCenterOfMass,
    "monopolar": LocalizeMonopolarTriangulation,
    "grid": LocalizeGridConvolution,
}