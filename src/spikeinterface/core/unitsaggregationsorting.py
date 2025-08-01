from __future__ import annotations

import math
import warnings
import numpy as np

from spikeinterface.core.core_tools import define_function_from_class
from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.basesorting import BaseSorting, BaseSortingSegment
from spikeinterface.core.segmentutils import _check_sampling_frequencies


class UnitsAggregationSorting(BaseSorting):
    """
    Aggregates units of multiple sortings into a single sorting object

    Parameters
    ----------
    sorting_list: list | dict
        List of BaseSorting objects to aggregate
    renamed_unit_ids: array-like
        If given, unit ids are renamed as provided. If None, unit ids are sequential integers.
    sampling_frequency_max_diff : float, default: 0
        Maximum allowed difference of sampling frequencies across recordings

    Returns
    -------
    aggregate_sorting: UnitsAggregationSorting
        The aggregated sorting object
    """

    def __init__(self, sorting_list, renamed_unit_ids=None, sampling_frequency_max_diff=0):
        unit_map = {}

        sorting_keys = []
        if isinstance(sorting_list, dict):
            sorting_keys = list(sorting_list.keys())
            sorting_list = list(sorting_list.values())

        num_all_units = sum([sort.get_num_units() for sort in sorting_list])
        if renamed_unit_ids is not None:
            assert len(np.unique(renamed_unit_ids)) == num_all_units, (
                "'renamed_unit_ids' doesn't have the right size" "or has duplicates!"
            )
            unit_ids = list(renamed_unit_ids)
        else:
            unit_ids_dtypes = [sort.get_unit_ids().dtype for sort in sorting_list]
            all_ids_are_same_type = np.unique(unit_ids_dtypes).size == 1
            all_units_ids_are_unique = False
            if all_ids_are_same_type:
                combined_ids = np.concatenate([sort.get_unit_ids() for sort in sorting_list])
                all_units_ids_are_unique = np.unique(combined_ids).size == num_all_units

            if all_ids_are_same_type and all_units_ids_are_unique:
                unit_ids = combined_ids
            else:
                default_unit_ids = [str(i) for i in range(num_all_units)]
                if all_ids_are_same_type and np.issubdtype(unit_ids_dtypes[0], np.integer):
                    unit_ids = np.arange(num_all_units, dtype=np.uint64)
                else:
                    unit_ids = default_unit_ids

        # unit map maps unit ids that are used to get spike trains
        u_id = 0
        for s_i, sorting in enumerate(sorting_list):
            single_unit_ids = sorting.get_unit_ids()
            for unit_id in single_unit_ids:
                unit_map[unit_ids[u_id]] = {"sorting_id": s_i, "unit_id": unit_id}
                u_id += 1

        sampling_frequencies = [sort.sampling_frequency for sort in sorting_list]
        num_segments = sorting_list[0].get_num_segments()

        _check_sampling_frequencies(sampling_frequencies, sampling_frequency_max_diff)
        sampling_frequency = sampling_frequencies[0]
        num_segments_ok = all(num_segments == sort.get_num_segments() for sort in sorting_list)
        if not num_segments_ok:
            raise ValueError("Sortings don't have the same num_segments")

        BaseSorting.__init__(self, sampling_frequency, unit_ids)

        annotation_keys = sorting_list[0].get_annotation_keys()
        for annotation_name in annotation_keys:
            if not all([annotation_name in sort.get_annotation_keys() for sort in sorting_list]):
                continue

            annotations = np.array([sort.get_annotation(annotation_name, copy=False) for sort in sorting_list])
            if np.all(annotations == annotations[0]):
                self.set_annotation(annotation_name, sorting_list[0].get_annotation(annotation_name))

        # Check if all the sortings have the same properties
        properties_set = set(np.concatenate([sorting.get_property_keys() for sorting in sorting_list]))
        for prop_name in properties_set:

            dtypes_per_sorting = []
            for sort in sorting_list:
                if prop_name in sort.get_property_keys():
                    dtypes_per_sorting.append(sort.get_property(prop_name).dtype.kind)

            if len(set(dtypes_per_sorting)) != 1:
                warnings.warn(
                    f"Skipping property '{prop_name}'. Difference in dtype.kind between sortings: {dtypes_per_sorting}"
                )
                continue

            all_property_values = []
            for sort in sorting_list:

                # If one of the sortings doesn't have the property, use the default missing property value
                if prop_name not in sort.get_property_keys():
                    try:
                        values = np.full(
                            sort.get_num_units(),
                            BaseExtractor.default_missing_property_values[dtypes_per_sorting[0]],
                        )
                    except:
                        warnings.warn(f"Skipping property '{prop_name}: cannot inpute missing property values.'")
                        break
                else:
                    values = sort.get_property(prop_name)

                all_property_values.append(values)

            try:
                prop_values = np.concatenate(all_property_values)
                self.set_property(key=prop_name, values=prop_values)
            except Exception as ext:
                warnings.warn(f"Skipping property '{prop_name}' as numpy cannot concatente. Numpy error: {ext}")

        # add a label to each unit, with which sorting it came from
        if len(sorting_keys) > 0:
            aggregation_keys = []
            for sort_key, sort in zip(sorting_keys, sorting_list):
                aggregation_keys += [sort_key] * sort.get_num_units()
            self.set_property(key="aggregation_key", values=aggregation_keys)

        # add segments
        for i_seg in range(num_segments):
            parent_segments = [sort._sorting_segments[i_seg] for sort in sorting_list]
            sub_segment = UnitsAggregationSortingSegment(unit_map, parent_segments)
            self.add_sorting_segment(sub_segment)

        self._sortings = sorting_list
        self._kwargs = {"sorting_list": sorting_list, "renamed_unit_ids": renamed_unit_ids}

    @property
    def sortings(self):
        return self._sortings


class UnitsAggregationSortingSegment(BaseSortingSegment):
    def __init__(self, unit_map, parent_segments):
        BaseSortingSegment.__init__(self)
        self._unit_map = unit_map
        self._parent_segments = parent_segments

    def get_unit_spike_train(
        self,
        unit_id,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> np.ndarray:
        sorting_id = self._unit_map[unit_id]["sorting_id"]
        unit_id_sorting = self._unit_map[unit_id]["unit_id"]
        times = self._parent_segments[sorting_id].get_unit_spike_train(unit_id_sorting, start_frame, end_frame)
        return times


aggregate_units = define_function_from_class(UnitsAggregationSorting, "aggregate_units")
