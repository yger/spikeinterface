from __future__ import annotations
from .uniform import UniformSelection
from .smart_sampling import SmartSamplingByAmplitudes

selection_methods = {
    "uniform": UniformSelection,
    "smart_sampling_by_amplitude": SmartSamplingByAmplitudes,
}