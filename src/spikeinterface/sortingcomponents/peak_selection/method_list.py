from __future__ import annotations
from .uniform import UniformSelection
from .smart_sampling import SmartSamplingByAmplitudes, SmartSamplingByLocations, SmartSamplingByLocationsAndTimes

selection_methods = {
    "uniform": UniformSelection,
    "smart_sampling_by_amplitude": SmartSamplingByAmplitudes,
    "smart_sampling_by_locations": SmartSamplingByLocations,
    "smart_sampling_by_locations_and_times": SmartSamplingByLocationsAndTimes,
}