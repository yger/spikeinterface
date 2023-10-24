import numpy as np

from spikeinterface.core.job_tools import _shared_job_kwargs_doc, fix_job_kwargs

from spikeinterface.core.template_tools import get_template_extremum_channel, get_template_extremum_channel_peak_shift

from spikeinterface.core.waveform_extractor import WaveformExtractor, BaseWaveformExtractorExtension
from spikeinterface.core.node_pipeline import SpikeRetriever


class SpikeLocationsCalculator(BaseWaveformExtractorExtension):
    """
    Computes spike locations from WaveformExtractor.

    Parameters
    ----------
    waveform_extractor: WaveformExtractor
        A waveform extractor object
    """

    extension_name = "spike_locations"

    def __init__(self, waveform_extractor):
        BaseWaveformExtractorExtension.__init__(self, waveform_extractor)

        extremum_channel_inds = get_template_extremum_channel(self.waveform_extractor, outputs="index")
        self.spikes = self.waveform_extractor.sorting.to_spike_vector(extremum_channel_inds=extremum_channel_inds)

    def _set_params(
        self, ms_before=0.5, ms_after=0.5, channel_from_template=True, method="center_of_mass", method_kwargs={}
    ):
        params = dict(
            ms_before=ms_before, ms_after=ms_after, channel_from_template=channel_from_template, method=method
        )
        params.update(**method_kwargs)
        return params

    def _select_extension_data(self, unit_ids):
        old_unit_ids = self.waveform_extractor.sorting.unit_ids
        unit_inds = np.flatnonzero(np.isin(old_unit_ids, unit_ids))

        spike_mask = np.isin(self.spikes["unit_index"], unit_inds)
        new_spike_locations = self._extension_data["spike_locations"][spike_mask]
        return dict(spike_locations=new_spike_locations)

    def _run(self, **job_kwargs):
        """
        This function first transforms the sorting object into a `peaks` numpy array and then
        uses the`sortingcomponents.peak_localization.localize_peaks()` function to triangulate
        spike locations.
        """
        from spikeinterface.sortingcomponents.peak_localization import _run_localization_from_peak_source

        job_kwargs = fix_job_kwargs(job_kwargs)

        we = self.waveform_extractor

        extremum_channel_inds = get_template_extremum_channel(we, peak_sign="neg", outputs="index")

        params = self._params.copy()
        channel_from_template = params.pop("channel_from_template")

        # @alessio @pierre: where do we expose the parameters of radius for the retriever (this is not the same as the one for locatization it is smaller) ???
        spike_retriever = SpikeRetriever(
            we.recording,
            we.sorting,
            channel_from_template=channel_from_template,
            extremum_channel_inds=extremum_channel_inds,
            radius_um=50,
            peak_sign=self._params.get("peaks_sign", "neg"),
        )
        spike_locations = _run_localization_from_peak_source(we.recording, spike_retriever, **params, **job_kwargs)

        self._extension_data["spike_locations"] = spike_locations

    def get_data(self, outputs="concatenated"):
        """
        Get computed spike locations

        Parameters
        ----------
        outputs : str, optional
            'concatenated' or 'by_unit', by default 'concatenated'

        Returns
        -------
        spike_locations : np.array or dict
            The spike locations as a structured array (outputs='concatenated') or
            as a dict with units as key and spike locations as values.
        """
        we = self.waveform_extractor
        sorting = we.sorting

        if outputs == "concatenated":
            return self._extension_data["spike_locations"]

        elif outputs == "by_unit":
            locations_by_unit = []
            for segment_index in range(self.waveform_extractor.get_num_segments()):
                i0 = np.searchsorted(self.spikes["segment_index"], segment_index, side="left")
                i1 = np.searchsorted(self.spikes["segment_index"], segment_index, side="right")
                spikes = self.spikes[i0:i1]
                locations = self._extension_data["spike_locations"][i0:i1]

                locations_by_unit.append({})
                for unit_ind, unit_id in enumerate(sorting.unit_ids):
                    mask = spikes["unit_index"] == unit_ind
                    locations_by_unit[segment_index][unit_id] = locations[mask]
            return locations_by_unit

    @staticmethod
    def get_extension_function():
        return compute_spike_locations


WaveformExtractor.register_extension(SpikeLocationsCalculator)

# @alessio @pierre: channel_from_template=True is the old behavior but this is not accurate
# what do we put by default ?


def compute_spike_locations(
    waveform_extractor,
    load_if_exists=False,
    ms_before=0.5,
    ms_after=0.5,
    channel_from_template=True,
    method="center_of_mass",
    method_kwargs={},
    outputs="concatenated",
    **job_kwargs,
):
    """
    Localize spikes in 2D or 3D with several methods given the template.

    Parameters
    ----------
    waveform_extractor : WaveformExtractor
        A waveform extractor object.
    load_if_exists : bool, default: False
        Whether to load precomputed spike locations, if they already exist.
    ms_before : float
        The left window, before a peak, in milliseconds.
    ms_after : float
        The right window, after a peak, in milliseconds.
    channel_from_template: bool, default True
        For each spike is the maximum channel computed from template or re estimated at every spikes.
        channel_from_template = True is old behavior but less acurate
        channel_from_template = False is slower but more accurate
    method : str
        'center_of_mass' / 'monopolar_triangulation' / 'grid_convolution'
    method_kwargs : dict
        Other kwargs depending on the method.
    outputs : str
        'concatenated' (default) / 'by_unit'
    {}

    Returns
    -------
    spike_locations: np.array or list of dict
        The spike locations.
            - If 'concatenated' all locations for all spikes and all units are concatenated
            - If 'by_unit', locations are returned as a list (for segments) of dictionaries (for units)
    """
    if load_if_exists and waveform_extractor.is_extension(SpikeLocationsCalculator.extension_name):
        slc = waveform_extractor.load_extension(SpikeLocationsCalculator.extension_name)
    else:
        slc = SpikeLocationsCalculator(waveform_extractor)
        slc.set_params(
            ms_before=ms_before,
            ms_after=ms_after,
            channel_from_template=channel_from_template,
            method=method,
            method_kwargs=method_kwargs,
        )
        slc.run(**job_kwargs)

    locs = slc.get_data(outputs=outputs)
    return locs


compute_spike_locations.__doc__.format(_shared_job_kwargs_doc)
