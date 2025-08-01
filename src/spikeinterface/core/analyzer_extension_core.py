"""
Implement AnalyzerExtension that are essential and imported in core
  * ComputeRandomSpikes
  * ComputeWaveforms
  * ComputeTemplates
Theses two classes replace the WaveformExtractor

It also implements:
  * ComputeNoiseLevels which is very convenient to have
"""

import warnings
import numpy as np

from .sortinganalyzer import AnalyzerExtension, register_result_extension
from .waveform_tools import extract_waveforms_to_single_buffer, estimate_templates_with_accumulator
from .recording_tools import get_noise_levels
from .template import Templates
from .sorting_tools import random_spikes_selection


class ComputeRandomSpikes(AnalyzerExtension):
    """
    AnalyzerExtension that select somes random spikes.
    This allows for a subsampling of spikes for further calculations and is important
    for managing that amount of memory and speed of computation in the analyzer.

    This will be used by the `waveforms`/`templates` extensions.

    This internally uses `random_spikes_selection()` parameters.

    Parameters
    ----------
    method : "uniform" | "all", default: "uniform"
        The method to select the spikes
    max_spikes_per_unit : int, default: 500
        The maximum number of spikes per unit, ignored if method="all"
    margin_size : int, default: None
        A margin on each border of segments to avoid border spikes, ignored if method="all"
    seed : int or None, default: None
        A seed for the random generator, ignored if method="all"

    Returns
    -------
    random_spike_indices: np.array
        The indices of the selected spikes
    """

    extension_name = "random_spikes"
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    need_job_kwargs = False

    def _run(self, verbose=False):

        self.data["random_spikes_indices"] = random_spikes_selection(
            self.sorting_analyzer.sorting,
            num_samples=self.sorting_analyzer.rec_attributes["num_samples"],
            **self.params,
        )

    def _set_params(self, method="uniform", max_spikes_per_unit=500, margin_size=None, seed=None):
        params = dict(method=method, max_spikes_per_unit=max_spikes_per_unit, margin_size=margin_size, seed=seed)
        return params

    def _select_extension_data(self, unit_ids):
        random_spikes_indices = self.data["random_spikes_indices"]

        spikes = self.sorting_analyzer.sorting.to_spike_vector()

        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))
        keep_spike_mask = np.isin(spikes["unit_index"], keep_unit_indices)

        selected_mask = np.zeros(spikes.size, dtype=bool)
        selected_mask[random_spikes_indices] = True

        new_data = dict()
        new_data["random_spikes_indices"] = np.flatnonzero(selected_mask[keep_spike_mask])
        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        new_data = dict()
        random_spikes_indices = self.data["random_spikes_indices"]
        if keep_mask is None:
            new_data["random_spikes_indices"] = random_spikes_indices.copy()
        else:
            spikes = self.sorting_analyzer.sorting.to_spike_vector()
            selected_mask = np.zeros(spikes.size, dtype=bool)
            selected_mask[random_spikes_indices] = True
            new_data["random_spikes_indices"] = np.flatnonzero(selected_mask[keep_mask])
        return new_data

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        new_data = dict()
        new_data["random_spikes_indices"] = self.data["random_spikes_indices"].copy()
        return new_data

    def _get_data(self):
        return self.data["random_spikes_indices"]

    def get_random_spikes(self):
        # utils to get the some_spikes vector
        # use internal cache
        if not hasattr(self, "_some_spikes"):
            spikes = self.sorting_analyzer.sorting.to_spike_vector()
            self._some_spikes = spikes[self.data["random_spikes_indices"]]
        return self._some_spikes

    def get_selected_indices_in_spike_train(self, unit_id, segment_index):
        # useful for WaveformExtractor backwards compatibility
        # In Waveforms extractor "selected_spikes" was a dict (key: unit_id) of list (segment_index) of indices of spikes in spiketrain
        sorting = self.sorting_analyzer.sorting
        random_spikes_indices = self.data["random_spikes_indices"]

        unit_index = sorting.id_to_index(unit_id)
        spikes = sorting.to_spike_vector()
        spike_indices_in_seg = np.flatnonzero(
            (spikes["segment_index"] == segment_index) & (spikes["unit_index"] == unit_index)
        )
        common_element, inds_left, inds_right = np.intersect1d(
            spike_indices_in_seg, random_spikes_indices, return_indices=True
        )
        selected_spikes_in_spike_train = inds_left
        return selected_spikes_in_spike_train


compute_random_spikes = ComputeRandomSpikes.function_factory()
register_result_extension(ComputeRandomSpikes)


class ComputeWaveforms(AnalyzerExtension):
    """
    AnalyzerExtension that extract some waveforms of each units.

    The sparsity is controlled by the SortingAnalyzer sparsity.

    Parameters
    ----------
    ms_before : float, default: 1.0
        The number of ms to extract before the spike events
    ms_after : float, default: 2.0
        The number of ms to extract after the spike events
    dtype : None | dtype, default: None
        The dtype of the waveforms. If None, the dtype of the recording is used.

    Returns
    -------
    waveforms : np.ndarray
        Array with computed waveforms with shape (num_random_spikes, num_samples, num_channels)
    """

    extension_name = "waveforms"
    depend_on = ["random_spikes"]
    need_recording = True
    use_nodepipeline = False
    need_job_kwargs = True

    @property
    def nbefore(self):
        return int(self.params["ms_before"] * self.sorting_analyzer.sampling_frequency / 1000.0)

    @property
    def nafter(self):
        return int(self.params["ms_after"] * self.sorting_analyzer.sampling_frequency / 1000.0)

    def _run(self, verbose=False, **job_kwargs):
        self.data.clear()

        recording = self.sorting_analyzer.recording
        sorting = self.sorting_analyzer.sorting
        unit_ids = sorting.unit_ids

        # retrieve spike vector and the sampling
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()

        if self.format == "binary_folder":
            # in that case waveforms are extacted directly in files
            file_path = self._get_binary_extension_folder() / "waveforms.npy"
            mode = "memmap"
            copy = False
        else:
            file_path = None
            mode = "shared_memory"
            copy = True

        if self.sparsity is None:
            sparsity_mask = None
        else:
            sparsity_mask = self.sparsity.mask

        all_waveforms = extract_waveforms_to_single_buffer(
            recording,
            some_spikes,
            unit_ids,
            self.nbefore,
            self.nafter,
            mode=mode,
            return_in_uV=self.sorting_analyzer.return_in_uV,
            file_path=file_path,
            dtype=self.params["dtype"],
            sparsity_mask=sparsity_mask,
            copy=copy,
            job_name="compute_waveforms",
            verbose=verbose,
            **job_kwargs,
        )

        self.data["waveforms"] = all_waveforms

    def _set_params(
        self,
        ms_before: float = 1.0,
        ms_after: float = 2.0,
        dtype=None,
    ):
        recording = self.sorting_analyzer.recording
        if dtype is None:
            dtype = recording.get_dtype()

        if np.issubdtype(dtype, np.integer) and self.sorting_analyzer.return_in_uV:
            dtype = "float32"

        dtype = np.dtype(dtype)

        params = dict(
            ms_before=float(ms_before),
            ms_after=float(ms_after),
            dtype=dtype.str,
        )
        return params

    def _select_extension_data(self, unit_ids):
        # random_spikes_indices = self.sorting_analyzer.get_extension("random_spikes").get_data()
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()

        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))
        spikes = self.sorting_analyzer.sorting.to_spike_vector()
        # some_spikes = spikes[random_spikes_indices]
        keep_spike_mask = np.isin(some_spikes["unit_index"], keep_unit_indices)

        new_data = dict()
        new_data["waveforms"] = self.data["waveforms"][keep_spike_mask, :, :]

        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        waveforms = self.data["waveforms"]
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()
        if keep_mask is not None:
            spike_indices = self.sorting_analyzer.get_extension("random_spikes").get_data()
            valid = keep_mask[spike_indices]
            some_spikes = some_spikes[valid]
            waveforms = waveforms[valid]
        else:
            waveforms = waveforms.copy()

        old_sparsity = self.sorting_analyzer.sparsity
        if old_sparsity is not None:
            # we need a realignement inside each group because we take the channel intersection sparsity
            for group_ids in merge_unit_groups:
                group_indices = self.sorting_analyzer.sorting.ids_to_indices(group_ids)
                group_sparsity_mask = old_sparsity.mask[group_indices, :]
                group_selection = []
                for unit_id in group_ids:
                    unit_index = self.sorting_analyzer.sorting.id_to_index(unit_id)
                    selection = np.flatnonzero(some_spikes["unit_index"] == unit_index)
                    group_selection.append(selection)
                _inplace_sparse_realign_waveforms(waveforms, group_selection, group_sparsity_mask)

            old_num_chans = int(np.max(np.sum(old_sparsity.mask, axis=1)))
            new_num_chans = int(np.max(np.sum(new_sorting_analyzer.sparsity.mask, axis=1)))
            if new_num_chans < old_num_chans:
                waveforms = waveforms[:, :, :new_num_chans]

        return dict(waveforms=waveforms)

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        # splitting only affects random spikes, not waveforms
        new_data = dict(waveforms=self.data["waveforms"].copy())
        return new_data

    def get_waveforms_one_unit(self, unit_id, force_dense: bool = False):
        """
        Returns the waveforms of a unit id.

        Parameters
        ----------
        unit_id : int or str
            The unit id to return waveforms for
        force_dense : bool, default: False
            If True, and SortingAnalyzer must be sparse then only waveforms on sparse channels are returned.

        Returns
        -------
        waveforms: np.array
            The waveforms (num_waveforms, num_samples, num_channels).
            In case sparsity is used, only the waveforms on sparse channels are returned.
        """
        sorting = self.sorting_analyzer.sorting
        unit_index = sorting.id_to_index(unit_id)

        waveforms = self.data["waveforms"]
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()

        spike_mask = some_spikes["unit_index"] == unit_index
        wfs = waveforms[spike_mask, :, :]

        if self.sorting_analyzer.sparsity is not None:
            chan_inds = self.sorting_analyzer.sparsity.unit_id_to_channel_indices[unit_id]
            wfs = wfs[:, :, : chan_inds.size]
            if force_dense:
                num_channels = self.sorting_analyzer.get_num_channels()
                dense_wfs = np.zeros((wfs.shape[0], wfs.shape[1], num_channels), dtype=wfs.dtype)
                dense_wfs[:, :, chan_inds] = wfs
                wfs = dense_wfs

        return wfs

    def _get_data(self):
        return self.data["waveforms"]


def _inplace_sparse_realign_waveforms(waveforms, group_selection, group_sparsity_mask):
    # this is used by "waveforms" extension but also "pca"

    # common mask is intersection
    common_mask = np.all(group_sparsity_mask, axis=0)

    for i in range(len(group_selection)):
        chan_mask = group_sparsity_mask[i, :]
        sel = group_selection[i]
        wfs = waveforms[sel, :, :][:, :, : np.sum(chan_mask)]
        keep_mask = common_mask[chan_mask]
        wfs = wfs[:, :, keep_mask]
        waveforms[:, :, : wfs.shape[2]][sel, :, :] = wfs
        waveforms[:, :, wfs.shape[2] :][sel, :, :] = 0.0


compute_waveforms = ComputeWaveforms.function_factory()
register_result_extension(ComputeWaveforms)


class ComputeTemplates(AnalyzerExtension):
    """
    AnalyzerExtension that computes templates (average, std, median, percentile, ...)

    This depends on the "waveforms" extension (`SortingAnalyzer.compute("waveforms")`)

    When the "waveforms" extension is already computed, then the recording is not needed anymore for this extension.

    Note: by default only the average and std are computed. Other operators (std, median, percentile) can be computed on demand
    after the SortingAnalyzer.compute("templates") and then the data dict is updated on demand.

    Parameters
    ----------
    operators: list[str] | list[(str, float)] (for percentile)
        The operators to compute. Can be "average", "std", "median", "percentile"
        If percentile is used, then the second element of the tuple is the percentile to compute.

    Returns
    -------
    templates: np.ndarray
        The computed templates with shape (num_units, num_samples, num_channels)
    """

    extension_name = "templates"
    depend_on = ["random_spikes|waveforms"]
    need_recording = True
    use_nodepipeline = False
    need_job_kwargs = True
    need_backward_compatibility_on_load = True

    def _handle_backward_compatibility_on_load(self):
        if "ms_before" not in self.params:
            # compatibility february 2024 > july 2024
            self.params["ms_before"] = self.params["nbefore"] * 1000.0 / self.sorting_analyzer.sampling_frequency

        if "ms_after" not in self.params:
            # compatibility february 2024 > july 2024
            self.params["ms_after"] = self.params["nafter"] * 1000.0 / self.sorting_analyzer.sampling_frequency

    def _set_params(self, ms_before: float = 1.0, ms_after: float = 2.0, operators=None):
        operators = operators or ["average", "std"]
        assert isinstance(operators, list)
        for operator in operators:
            if isinstance(operator, str):
                if operator not in ("average", "std", "median", "mad"):
                    error_msg = (
                        f"You have entered an operator {operator} in your `operators` argument which is "
                        f"not supported. Please use any of ['average', 'std', 'median', 'mad'] instead."
                    )
                    raise ValueError(error_msg)
            else:
                assert isinstance(operator, (list, tuple))
                assert len(operator) == 2
                assert operator[0] == "percentile"

        waveforms_extension = self.sorting_analyzer.get_extension("waveforms")
        if waveforms_extension is not None:
            ms_before = waveforms_extension.params["ms_before"]
            ms_after = waveforms_extension.params["ms_after"]

        params = dict(
            operators=operators,
            ms_before=ms_before,
            ms_after=ms_after,
        )
        return params

    def _run(self, verbose=False, **job_kwargs):
        self.data.clear()

        if self.sorting_analyzer.has_extension("waveforms"):
            self._compute_and_append_from_waveforms(self.params["operators"])

        else:
            bad_operator_list = [
                operator for operator in self.params["operators"] if operator not in ("average", "std")
            ]
            if len(bad_operator_list) > 0:
                raise ValueError(
                    f"Computing templates with operators {bad_operator_list} requires the 'waveforms' extension"
                )

            recording = self.sorting_analyzer.recording
            sorting = self.sorting_analyzer.sorting
            unit_ids = sorting.unit_ids

            # retrieve spike vector and the sampling
            some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()

            return_in_uV = self.sorting_analyzer.return_in_uV

            return_std = "std" in self.params["operators"]
            output = estimate_templates_with_accumulator(
                recording,
                some_spikes,
                unit_ids,
                self.nbefore,
                self.nafter,
                return_in_uV=return_in_uV,
                return_std=return_std,
                verbose=verbose,
                **job_kwargs,
            )

            # Output of estimate_templates_with_accumulator is either (templates,) or (templates, stds)
            if return_std:
                templates, stds = output
                self.data["average"] = templates
                self.data["std"] = stds
            else:
                self.data["average"] = output

    def _compute_and_append_from_waveforms(self, operators):
        if not self.sorting_analyzer.has_extension("waveforms"):
            raise ValueError(f"Computing templates with operators {operators} requires the 'waveforms' extension")

        unit_ids = self.sorting_analyzer.unit_ids
        channel_ids = self.sorting_analyzer.channel_ids
        waveforms_extension = self.sorting_analyzer.get_extension("waveforms")
        waveforms = waveforms_extension.data["waveforms"]

        num_samples = waveforms.shape[1]

        for operator in operators:
            if isinstance(operator, str) and operator in ("average", "std", "median"):
                key = operator
            elif isinstance(operator, (list, tuple)):
                operator, percentile = operator
                assert operator == "percentile"
                key = f"pencentile_{percentile}"
            else:
                raise ValueError(f"ComputeTemplates: wrong operator {operator}")
            self.data[key] = np.zeros((unit_ids.size, num_samples, channel_ids.size))

        # spikes = self.sorting_analyzer.sorting.to_spike_vector()
        # some_spikes = spikes[self.sorting_analyzer.random_spikes_indices]

        assert self.sorting_analyzer.has_extension(
            "random_spikes"
        ), "compute 'templates' requires the random_spikes extension. You can run sorting_analyzer.compute('random_spikes')"
        some_spikes = self.sorting_analyzer.get_extension("random_spikes").get_random_spikes()
        for unit_index, unit_id in enumerate(unit_ids):
            spike_mask = some_spikes["unit_index"] == unit_index
            wfs = waveforms[spike_mask, :, :]
            if wfs.shape[0] == 0:
                continue

            for operator in operators:
                if operator == "average":
                    arr = np.average(wfs, axis=0)
                    key = operator
                elif operator == "std":
                    arr = np.std(wfs, axis=0)
                    key = operator
                elif operator == "median":
                    arr = np.median(wfs, axis=0)
                    key = operator
                elif isinstance(operator, (list, tuple)):
                    operator, percentile = operator
                    arr = np.percentile(wfs, percentile, axis=0)
                    key = f"pencentile_{percentile}"

                if self.sparsity is None:
                    self.data[key][unit_index, :, :] = arr
                else:
                    channel_indices = self.sparsity.unit_id_to_channel_indices[unit_id]
                    self.data[key][unit_index, :, :][:, channel_indices] = arr[:, : channel_indices.size]

    @property
    def nbefore(self):
        nbefore = int(self.params["ms_before"] * self.sorting_analyzer.sampling_frequency / 1000.0)
        return nbefore

    @property
    def nafter(self):
        nafter = int(self.params["ms_after"] * self.sorting_analyzer.sampling_frequency / 1000.0)
        return nafter

    def _select_extension_data(self, unit_ids):
        keep_unit_indices = np.flatnonzero(np.isin(self.sorting_analyzer.unit_ids, unit_ids))

        new_data = dict()
        for key, arr in self.data.items():
            new_data[key] = arr[keep_unit_indices, :, :]

        return new_data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):

        all_new_units = new_sorting_analyzer.unit_ids
        new_data = dict()
        counts = self.sorting_analyzer.sorting.count_num_spikes_per_unit()
        for key, arr in self.data.items():
            new_data[key] = np.zeros((len(all_new_units), arr.shape[1], arr.shape[2]), dtype=arr.dtype)
            for unit_index, unit_id in enumerate(all_new_units):
                if unit_id not in new_unit_ids:
                    keep_unit_index = self.sorting_analyzer.sorting.id_to_index(unit_id)
                    new_data[key][unit_index] = arr[keep_unit_index, :, :]
                else:
                    merge_group = merge_unit_groups[list(new_unit_ids).index(unit_id)]
                    keep_unit_indices = self.sorting_analyzer.sorting.ids_to_indices(merge_group)
                    # We do a weighted sum of the templates
                    weights = np.zeros(len(merge_group), dtype=np.float32)
                    for count, merge_unit_id in enumerate(merge_group):
                        weights[count] = counts[merge_unit_id]
                    weights /= weights.sum()
                    new_data[key][unit_index] = (arr[keep_unit_indices, :, :] * weights[:, np.newaxis, np.newaxis]).sum(
                        0
                    )
                    if new_sorting_analyzer.sparsity is not None:
                        chan_ids = new_sorting_analyzer.sparsity.unit_id_to_channel_indices[unit_id]
                        mask = ~np.isin(np.arange(arr.shape[2]), chan_ids)
                        new_data[key][unit_index][:, mask] = 0

        return new_data

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        if not new_sorting_analyzer.has_extension("waveforms"):
            warnings.warn(
                "Splitting templates without the 'waveforms' extension will simply copy the template of the unit that "
                "was split to the new split units. This is not recommended and may lead to incorrect results. It is "
                "recommended to compute the 'waveforms' extension before splitting, or to use 'hard' splitting mode.",
            )
        new_data = dict()
        for operator, arr in self.data.items():
            # we first copy the unsplit units
            new_array = np.zeros((len(new_sorting_analyzer.unit_ids), arr.shape[1], arr.shape[2]), dtype=arr.dtype)
            new_analyzer_unit_ids = list(new_sorting_analyzer.unit_ids)
            unsplit_unit_ids = [unit_id for unit_id in self.sorting_analyzer.unit_ids if unit_id not in split_units]
            new_indices = np.array([new_analyzer_unit_ids.index(unit_id) for unit_id in unsplit_unit_ids])
            old_indices = self.sorting_analyzer.sorting.ids_to_indices(unsplit_unit_ids)
            new_array[new_indices, ...] = arr[old_indices, ...]

            for split_unit_id, new_splits in zip(split_units, new_unit_ids):
                if new_sorting_analyzer.has_extension("waveforms"):
                    for new_unit_id in new_splits:
                        split_unit_index = new_sorting_analyzer.sorting.id_to_index(new_unit_id)
                        wfs = new_sorting_analyzer.get_extension("waveforms").get_waveforms_one_unit(
                            new_unit_id, force_dense=True
                        )

                        if operator == "average":
                            arr = np.average(wfs, axis=0)
                        elif operator == "std":
                            arr = np.std(wfs, axis=0)
                        elif operator == "median":
                            arr = np.median(wfs, axis=0)
                        elif "percentile" in operator:
                            _, percentile = operator.splot("_")
                            arr = np.percentile(wfs, float(percentile), axis=0)
                        new_array[split_unit_index, ...] = arr
                else:
                    split_unit_index = self.sorting_analyzer.sorting.id_to_index(split_unit_id)
                    old_template = arr[split_unit_index, ...]
                    new_indices = new_sorting_analyzer.sorting.ids_to_indices(new_splits)
                    new_array[new_indices, ...] = np.tile(old_template, (len(new_splits), 1, 1))
            new_data[operator] = new_array
        return new_data

    def _get_data(self, operator="average", percentile=None, outputs="numpy"):
        if operator != "percentile":
            key = operator
        else:
            assert percentile is not None, "You must provide percentile=... if `operator=percentile`"
            key = f"percentile_{percentile}"

        if key not in self.data.keys():
            error_msg = (
                f"You have entered `operator={key}`, but the only operators calculated are "
                f"{list(self.data.keys())}. Please use one of these as your `operator` in the "
                f"`get_data` function."
            )
            raise ValueError(error_msg)

        templates_array = self.data[key]

        if outputs == "numpy":
            return templates_array
        elif outputs == "Templates":
            return Templates(
                templates_array=templates_array,
                sampling_frequency=self.sorting_analyzer.sampling_frequency,
                nbefore=self.nbefore,
                channel_ids=self.sorting_analyzer.channel_ids,
                unit_ids=self.sorting_analyzer.unit_ids,
                probe=self.sorting_analyzer.get_probe(),
            )
        else:
            raise ValueError("outputs must be `numpy` or `Templates`")

    def get_templates(self, unit_ids=None, operator="average", percentile=None, save=True, outputs="numpy"):
        """
        Return templates (average, std, median or percentiles) for multiple units.

        If not computed yet then this is computed on demand and optionally saved.

        Parameters
        ----------
        unit_ids : list or None
            Unit ids to retrieve waveforms for
        operator : "average" | "median" | "std" | "percentile", default: "average"
            The operator to compute the templates
        percentile : float, default: None
            Percentile to use for operator="percentile"
        save : bool, default: True
            In case, the operator is not computed yet it can be saved to folder or zarr
        outputs : "numpy" | "Templates", default: "numpy"
            Whether to return a numpy array or a Templates object

        Returns
        -------
        templates : np.array | Templates
            The returned templates (num_units, num_samples, num_channels)
        """
        if operator != "percentile":
            key = operator
        else:
            assert percentile is not None, "You must provide percentile=... if `operator='percentile'`"
            key = f"pencentile_{percentile}"

        if key in self.data:
            templates_array = self.data[key]
        else:
            if operator != "percentile":
                self._compute_and_append_from_waveforms([operator])
                self.params["operators"] += [operator]
            else:
                self._compute_and_append_from_waveforms([(operator, percentile)])
                self.params["operators"] += [(operator, percentile)]
            templates_array = self.data[key]

            if save:
                if not self.sorting_analyzer.is_read_only():
                    self.save()

        if unit_ids is not None:
            unit_indices = self.sorting_analyzer.sorting.ids_to_indices(unit_ids)
            templates_array = templates_array[unit_indices, :, :]
        else:
            unit_ids = self.sorting_analyzer.unit_ids

        if outputs == "numpy":
            return templates_array
        elif outputs == "Templates":
            return Templates(
                templates_array=templates_array,
                sampling_frequency=self.sorting_analyzer.sampling_frequency,
                nbefore=self.nbefore,
                channel_ids=self.sorting_analyzer.channel_ids,
                unit_ids=unit_ids,
                probe=self.sorting_analyzer.get_probe(),
                is_in_uV=self.sorting_analyzer.return_in_uV,
            )
        else:
            raise ValueError("`outputs` must be 'numpy' or 'Templates'")

    def get_unit_template(self, unit_id, operator="average"):
        """
        Return template for a single unit.

        Parameters
        ----------
        unit_id: str | int
            Unit id to retrieve waveforms for
        operator: str, default: "average"
             The operator to compute the templates

        Returns
        -------
        template: np.array
            The returned template (num_samples, num_channels)
        """

        templates = self.data[operator]
        unit_index = self.sorting_analyzer.sorting.id_to_index(unit_id)

        return np.array(templates[unit_index, :, :])


compute_templates = ComputeTemplates.function_factory()
register_result_extension(ComputeTemplates)


class ComputeNoiseLevels(AnalyzerExtension):
    """
    Computes the noise level associated with each recording channel.

    This function will wraps the `get_noise_levels(recording)` to make the noise levels persistent
    on disk (folder or zarr) as a `WaveformExtension`.
    The noise levels do not depend on the unit list, only the recording, but it is a convenient way to
    retrieve the noise levels directly ine the WaveformExtractor.

    Note that the noise levels can be scaled or not, depending on the `return_in_uV` parameter
    of the `SortingAnalyzer`.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        A SortingAnalyzer object
    **kwargs : dict
        Additional parameters for the `spikeinterface.get_noise_levels()` function

    Returns
    -------
    noise_levels : np.array
        The noise level vector
    """

    extension_name = "noise_levels"
    depend_on = []
    need_recording = True
    use_nodepipeline = False
    need_job_kwargs = True
    need_backward_compatibility_on_load = True

    def __init__(self, sorting_analyzer):
        AnalyzerExtension.__init__(self, sorting_analyzer)

    def _set_params(self, **noise_level_params):
        params = noise_level_params.copy()
        return params

    def _select_extension_data(self, unit_ids):
        # this does not depend on units
        return self.data

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask=None, verbose=False, **job_kwargs
    ):
        # this does not depend on units
        return self.data.copy()

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        # this does not depend on units
        return self.data.copy()

    def _run(self, verbose=False, **job_kwargs):
        self.data["noise_levels"] = get_noise_levels(
            self.sorting_analyzer.recording,
            return_in_uV=self.sorting_analyzer.return_in_uV,
            **self.params,
            **job_kwargs,
        )

    def _get_data(self):
        return self.data["noise_levels"]

    def _handle_backward_compatibility_on_load(self):
        # The old parameters used to be params=dict(num_chunks_per_segment=20, chunk_size=10000, seed=None)
        # now it is handle more explicitly using random_slices_kwargs=dict()
        for key in ("num_chunks_per_segment", "chunk_size", "seed"):
            if key in self.params:
                if "random_slices_kwargs" not in self.params:
                    self.params["random_slices_kwargs"] = dict()
                self.params["random_slices_kwargs"][key] = self.params.pop(key)


register_result_extension(ComputeNoiseLevels)
compute_noise_levels = ComputeNoiseLevels.function_factory()
