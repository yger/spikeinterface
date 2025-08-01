from __future__ import annotations
from typing import Literal, Optional, Any

from pathlib import Path
from itertools import chain
import os
import json
import math
import pickle
import weakref
import shutil
import warnings
import importlib
from copy import copy
from packaging.version import parse
from time import perf_counter

import numpy as np

import probeinterface

import spikeinterface

from spikeinterface.core import BaseRecording, BaseSorting, aggregate_channels, aggregate_units

from .recording_tools import check_probe_do_not_overlap, get_rec_attributes, do_recording_attributes_match
from .core_tools import (
    check_json,
    retrieve_importing_provenance,
    is_path_remote,
    clean_zarr_folder_name,
)
from .sorting_tools import (
    generate_unit_ids_for_merge_group,
    generate_unit_ids_for_split,
    check_unit_splits_consistency,
    _get_ids_after_merging,
    _get_ids_after_splitting,
)
from .job_tools import split_job_kwargs
from .numpyextractors import NumpySorting
from .sparsity import ChannelSparsity, estimate_sparsity
from .sortingfolder import NumpyFolderSorting
from .zarrextractors import get_default_zarr_compressor, ZarrSortingExtractor, super_zarr_open
from .node_pipeline import run_node_pipeline


# high level function
def create_sorting_analyzer(
    sorting,
    recording,
    format="memory",
    folder=None,
    sparse=True,
    sparsity=None,
    set_sparsity_by_dict_key=False,
    return_scaled=None,
    return_in_uV=True,
    overwrite=False,
    backend_options=None,
    **sparsity_kwargs,
) -> "SortingAnalyzer":
    """
    Create a SortingAnalyzer by pairing a Sorting and the corresponding Recording.

    This object will handle a list of AnalyzerExtension for all the post processing steps like: waveforms,
    templates, unit locations, spike locations, quality metrics ...

    This object will be also use used for plotting purpose.


    Parameters
    ----------
    sorting : Sorting | dict
        The sorting object, or a dict of them
    recording : Recording | dict
        The recording object, or a dict of them
    folder : str or Path or None, default: None
        The folder where analyzer is cached
    format : "memory | "binary_folder" | "zarr", default: "memory"
        The mode to store analyzer. If "folder", the analyzer is stored on disk in the specified folder.
        The "folder" argument must be specified in case of mode "folder".
        If "memory" is used, the analyzer is stored in RAM. Use this option carefully!
    sparse : bool, default: True
        If True, then a sparsity mask is computed using the `estimate_sparsity()` function using
        a few spikes to get an estimate of dense templates to create a ChannelSparsity object.
        Then, the sparsity will be propagated to all ResultExtention that handle sparsity (like wavforms, pca, ...)
        You can control `estimate_sparsity()` : all extra arguments are propagated to it (included job_kwargs)
    sparsity : ChannelSparsity or None, default: None
        The sparsity used to compute exensions. If this is given, `sparse` is ignored.
    set_sparsity_by_dict_key : bool, default: False
        If True and passing recording and sorting dicts, will set the sparsity based on the dict keys,
        and other `sparsity_kwargs` are overwritten. If False, use other sparsity settings.
    return_scaled : bool | None, default: None
        DEPRECATED. Use return_in_uV instead.
        All extensions that play with traces will use this global return_in_uV : "waveforms", "noise_levels", "templates".
        This prevent return_in_uV being differents from different extensions and having wrong snr for instance.
    return_in_uV : bool, default: None
        If True, all extensions that play with traces will use this global return_in_uV : "waveforms", "noise_levels", "templates".
        This prevent return_in_uV being differents from different extensions and having wrong snr for instance.
        If None, use return_scaled value.
    overwrite: bool, default: False
        If True, overwrite the folder if it already exists.
    backend_options : dict | None, default: None
        Keyword arguments for the backend specified by format. It can contain the:

            * storage_options: dict | None (fsspec storage options)
            * saving_options: dict | None (additional saving options for creating and saving datasets, e.g. compression/filters for zarr)

    sparsity_kwargs : keyword arguments

    Returns
    -------
    sorting_analyzer : SortingAnalyzer
        The SortingAnalyzer object

    Examples
    --------
    >>> import spikeinterface as si

    >>> # Create dense analyzer and save to disk with binary_folder format.
    >>> sorting_analyzer = si.create_sorting_analyzer(sorting, recording, format="binary_folder", folder="/path/to_my/result")

    >>> # Can be reload
    >>> sorting_analyzer = si.load_sorting_analyzer(folder="/path/to_my/result")

    >>> # Can run extension
    >>> sorting_analyzer = si.compute("unit_locations", ...)

    >>> # Can be copy to another format (extensions are propagated)
    >>> sorting_analyzer2 = sorting_analyzer.save_as(format="memory")
    >>> sorting_analyzer3 = sorting_analyzer.save_as(format="zarr", folder="/path/to_my/result.zarr")

    >>> # Can make a copy with a subset of units (extensions are propagated for the unit subset)
    >>> sorting_analyzer4 = sorting_analyzer.select_units(unit_ids=sorting.units_ids[:5], format="memory")
    >>> sorting_analyzer5 = sorting_analyzer.select_units(unit_ids=sorting.units_ids[:5], format="binary_folder", folder="/result_5units")

    Notes
    -----

    By default creating a SortingAnalyzer can be slow because the sparsity is estimated by default.
    In some situation, sparsity is not needed, so to make it fast creation, you need to turn
    sparsity off (or give external sparsity) like this.
    """

    if isinstance(sorting, dict) and isinstance(recording, dict):

        if sorting.keys() != recording.keys():
            raise ValueError(
                f"Keys of `sorting`, {sorting.keys()}, and `recording`, {recording.keys()}, dicts do not match."
            )

        aggregated_recording = aggregate_channels(recording)
        aggregated_sorting = aggregate_units(sorting)

        if set_sparsity_by_dict_key:
            sparsity_kwargs = {"method": "by_property", "by_property": "aggregation_key"}

        return create_sorting_analyzer(
            sorting=aggregated_sorting,
            recording=aggregated_recording,
            format=format,
            folder=folder,
            sparse=sparse,
            sparsity=sparsity,
            return_scaled=return_scaled,
            return_in_uV=return_in_uV,
            overwrite=overwrite,
            backend_options=backend_options,
            **sparsity_kwargs,
        )

    if format != "memory":
        if format == "zarr":
            if not is_path_remote(folder):
                folder = clean_zarr_folder_name(folder)
        if not is_path_remote(folder):
            if Path(folder).is_dir():
                if not overwrite:
                    raise ValueError(f"Folder already exists {folder}! Use overwrite=True to overwrite it.")
                else:
                    shutil.rmtree(folder)

    # handle sparsity
    if sparsity is not None:
        # some checks
        assert isinstance(sparsity, ChannelSparsity), "'sparsity' must be a ChannelSparsity object"
        assert np.array_equal(
            sorting.unit_ids, sparsity.unit_ids
        ), "create_sorting_analyzer(): if external sparsity is given unit_ids must correspond"
        assert np.array_equal(
            recording.channel_ids, sparsity.channel_ids
        ), "create_sorting_analyzer(): if external sparsity is given unit_ids must correspond"
    elif sparse:
        sparsity = estimate_sparsity(sorting, recording, **sparsity_kwargs)
    else:
        sparsity = None

    # Handle deprecated return_scaled parameter
    if return_scaled is not None:
        warnings.warn(
            "`return_scaled` is deprecated and will be removed in version 0.105.0. Use `return_in_uV` instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        return_in_uV = return_scaled

    if return_in_uV and not recording.has_scaleable_traces() and recording.get_dtype().kind == "i":
        print("create_sorting_analyzer: recording does not have scaling to uV, forcing return_in_uV=False")
        return_in_uV = False

    sorting_analyzer = SortingAnalyzer.create(
        sorting,
        recording,
        format=format,
        folder=folder,
        sparsity=sparsity,
        return_in_uV=return_in_uV,
        backend_options=backend_options,
    )

    return sorting_analyzer


def load_sorting_analyzer(folder, load_extensions=True, format="auto", backend_options=None) -> "SortingAnalyzer":
    """
    Load a SortingAnalyzer object from disk.

    Parameters
    ----------
    folder : str or Path
        The folder / zarr folder where the analyzer is stored. If the folder is a remote path stored in the cloud,
        the backend_options can be used to specify credentials. If the remote path is not accessible,
        and backend_options is not provided, the function will try to load the object in anonymous mode (anon=True),
        which enables to load data from open buckets.
    load_extensions : bool, default: True
        Load all extensions or not.
    format : "auto" | "binary_folder" | "zarr"
        The format of the folder.
    backend_options : dict | None, default: None
        The backend options for the backend.
        The dictionary can contain the following keys:

            * storage_options: dict | None (fsspec storage options)
            * saving_options: dict | None (additional saving options for creating and saving datasets)

    Returns
    -------
    sorting_analyzer : SortingAnalyzer
        The loaded SortingAnalyzer

    """
    return SortingAnalyzer.load(folder, load_extensions=load_extensions, format=format, backend_options=backend_options)


class SortingAnalyzer:
    """
    Class to make a pair of Recording-Sorting which will be used used for all post postprocessing,
    visualization and quality metric computation.

    This internally maintains a list of computed ResultExtention (waveform, pca, unit position, spike position, ...).

    This can live in memory and/or can be be persistent to disk in 2 internal formats (folder/json/npz or zarr).
    A SortingAnalyzer can be transfer to another format using `save_as()`

    This handle unit sparsity that can be propagated to ResultExtention.

    This handle spike sampling that can be propagated to ResultExtention : works on only a subset of spikes.

    This internally saves a copy of the Sorting and extracts main recording attributes (without traces) so
    the SortingAnalyzer object can be reloaded even if references to the original sorting and/or to the original recording
    are lost.

    SortingAnalyzer() should not never be used directly for creating: use instead create_sorting_analyzer(sorting, resording, ...)
    or eventually SortingAnalyzer.create(...)
    """

    def __init__(
        self,
        sorting: BaseSorting,
        recording: BaseRecording | None = None,
        rec_attributes: dict | None = None,
        format: str | None = None,
        sparsity: ChannelSparsity | None = None,
        return_in_uV: bool = True,
        backend_options: dict | None = None,
    ):
        # very fast init because checks are done in load and create
        self.sorting = sorting
        # self.recording will be a property
        self._recording = recording
        self.rec_attributes = rec_attributes
        self.format = format
        self.sparsity = sparsity
        self.return_in_uV = return_in_uV

        # For backward compatibility
        self.return_scaled = return_in_uV
        self.folder: str | Path | None = None

        # this is used to store temporary recording
        self._temporary_recording = None

        # backend-specific kwargs for different formats, which can be used to
        # set some parameters for saving (e.g., compression)
        #
        # - storage_options: dict | None (fsspec storage options)
        # - saving_options: dict | None
        # (additional saving options for creating and saving datasets, e.g. compression/filters for zarr)
        self._backend_options = {} if backend_options is None else backend_options

        # extensions are not loaded at init
        self.extensions = dict()

    def __repr__(self) -> str:
        clsname = self.__class__.__name__
        nseg = self.get_num_segments()
        nchan = self.get_num_channels()
        nunits = self.get_num_units()
        txt = f"{clsname}: {nchan} channels - {nunits} units - {nseg} segments - {self.format}"
        if self.format != "memory":
            if is_path_remote(str(self.folder)):
                txt += f" (remote)"
        if self.is_sparse():
            txt += " - sparse"
        if self.has_recording():
            txt += " - has recording"
        if self.has_temporary_recording():
            txt += " - has temporary recording"
        ext_txt = f"Loaded {len(self.extensions)} extensions"
        if len(self.extensions) > 0:
            ext_txt += f": {', '.join(self.extensions.keys())}"
        txt += "\n" + ext_txt
        return txt

    ## create and load zone

    @classmethod
    def create(
        cls,
        sorting: BaseSorting,
        recording: BaseRecording,
        format: Literal[
            "memory",
            "binary_folder",
            "zarr",
        ] = "memory",
        folder=None,
        sparsity=None,
        return_scaled=None,
        return_in_uV=True,
        backend_options=None,
    ):
        assert recording is not None, "To create a SortingAnalyzer you need to specify the recording"
        # some checks
        if sorting.sampling_frequency != recording.sampling_frequency:
            if math.isclose(sorting.sampling_frequency, recording.sampling_frequency, abs_tol=1e-2, rel_tol=1e-5):
                warnings.warn(
                    "Sorting and Recording have a small difference in sampling frequency. "
                    "This could be due to rounding of floats. Using the sampling frequency from the Recording."
                )
                # we make a copy here to change the smapling frequency
                sorting = NumpySorting.from_sorting(sorting, with_metadata=True, copy_spike_vector=True)
                sorting._sampling_frequency = recording.sampling_frequency
            else:
                raise ValueError(
                    f"Sorting and Recording sampling frequencies are too different: "
                    f"recording: {recording.sampling_frequency} - sorting: {sorting.sampling_frequency}. "
                    "Ensure that you are associating the correct Recording and Sorting when creating a SortingAnalyzer."
                )
        # check that multiple probes are non-overlapping
        all_probes = recording.get_probegroup().probes
        check_probe_do_not_overlap(all_probes)

        if format == "memory":
            sorting_analyzer = cls.create_memory(sorting, recording, sparsity, return_in_uV, rec_attributes=None)
        elif format == "binary_folder":
            sorting_analyzer = cls.create_binary_folder(
                folder,
                sorting,
                recording,
                sparsity,
                return_in_uV,
                rec_attributes=None,
                backend_options=backend_options,
            )
        elif format == "zarr":
            assert folder is not None, "For format='zarr' folder must be provided"
            if not is_path_remote(folder):
                folder = clean_zarr_folder_name(folder)
            sorting_analyzer = cls.create_zarr(
                folder,
                sorting,
                recording,
                sparsity,
                return_in_uV,
                rec_attributes=None,
                backend_options=backend_options,
            )
        else:
            raise ValueError("SortingAnalyzer.create: wrong format")

        return sorting_analyzer

    @classmethod
    def load(cls, folder, recording=None, load_extensions=True, format="auto", backend_options=None):
        """
        Load folder or zarr.
        The recording can be given if the recording location has changed.
        Otherwise the recording is loaded when possible.
        """
        if format == "auto":
            # make better assumption and check for auto guess format
            if Path(folder).suffix == ".zarr":
                format = "zarr"
            else:
                format = "binary_folder"

        if format == "binary_folder":
            sorting_analyzer = SortingAnalyzer.load_from_binary_folder(
                folder, recording=recording, backend_options=backend_options
            )
        elif format == "zarr":
            sorting_analyzer = SortingAnalyzer.load_from_zarr(
                folder, recording=recording, backend_options=backend_options
            )

        if not is_path_remote(str(folder)):
            if load_extensions:
                sorting_analyzer.load_all_saved_extension()

        return sorting_analyzer

    @classmethod
    def create_memory(cls, sorting, recording, sparsity, return_in_uV, rec_attributes):
        # used by create and save_as

        if rec_attributes is None:
            assert recording is not None
            rec_attributes = get_rec_attributes(recording)
            rec_attributes["probegroup"] = recording.get_probegroup()
        else:
            # a copy is done to avoid shared dict between instances (which can block garbage collector)
            rec_attributes = rec_attributes.copy()

        # a copy of sorting is copied in memory for fast access
        sorting_copy = NumpySorting.from_sorting(sorting, with_metadata=True, copy_spike_vector=True)

        sorting_analyzer = SortingAnalyzer(
            sorting=sorting_copy,
            recording=recording,
            rec_attributes=rec_attributes,
            format="memory",
            sparsity=sparsity,
            return_in_uV=return_in_uV,
        )
        return sorting_analyzer

    @classmethod
    def create_binary_folder(cls, folder, sorting, recording, sparsity, return_in_uV, rec_attributes, backend_options):
        # used by create and save_as

        folder = Path(folder)
        if folder.is_dir():
            raise ValueError(f"Folder already exists {folder}")
        folder.mkdir(parents=True)

        info_file = folder / f"spikeinterface_info.json"
        info = dict(
            version=spikeinterface.__version__,
            dev_mode=spikeinterface.DEV_MODE,
            object="SortingAnalyzer",
        )
        with open(info_file, mode="w") as f:
            json.dump(check_json(info), f, indent=4)

        # save a copy of the sorting
        sorting.save(folder=folder / "sorting")

        if recording is not None:
            # save recording and sorting provenance
            if recording.check_serializability("json"):
                recording.dump(folder / "recording.json", relative_to=folder)
            elif recording.check_serializability("pickle"):
                recording.dump(folder / "recording.pickle", relative_to=folder)
            else:
                warnings.warn("The Recording is not serializable! The recording link will be lost for future load")
        else:
            assert rec_attributes is not None, "recording or rec_attributes must be provided"
            warnings.warn("Recording not provided, instntiating SortingAnalyzer in recordingless mode.")

        if sorting.check_serializability("json"):
            sorting.dump(folder / "sorting_provenance.json", relative_to=folder)
        elif sorting.check_serializability("pickle"):
            sorting.dump(folder / "sorting_provenance.pickle", relative_to=folder)
        else:
            warnings.warn(
                "The sorting provenance is not serializable! The sorting provenance link will be lost for future load"
            )

        # dump recording attributes
        probegroup = None
        rec_attributes_file = folder / "recording_info" / "recording_attributes.json"
        rec_attributes_file.parent.mkdir()
        if rec_attributes is None:
            rec_attributes = get_rec_attributes(recording)
            rec_attributes_file.write_text(json.dumps(check_json(rec_attributes), indent=4), encoding="utf8")
            probegroup = recording.get_probegroup()
        else:
            rec_attributes_copy = rec_attributes.copy()
            probegroup = rec_attributes_copy.pop("probegroup")
            rec_attributes_file.write_text(json.dumps(check_json(rec_attributes_copy), indent=4), encoding="utf8")

        if probegroup is not None:
            probegroup_file = folder / "recording_info" / "probegroup.json"
            probeinterface.write_probeinterface(probegroup_file, probegroup)

        if sparsity is not None:
            np.save(folder / "sparsity_mask.npy", sparsity.mask)

        settings_file = folder / f"settings.json"
        settings = dict(
            return_in_uV=return_in_uV,
        )
        with open(settings_file, mode="w") as f:
            json.dump(check_json(settings), f, indent=4)

        return cls.load_from_binary_folder(folder, recording=recording, backend_options=backend_options)

    @classmethod
    def load_from_binary_folder(cls, folder, recording=None, backend_options=None):
        from .loading import load

        folder = Path(folder)
        assert folder.is_dir(), f"This folder does not exists {folder}"

        # load internal sorting copy in memory
        sorting = NumpySorting.from_sorting(
            NumpyFolderSorting(folder / "sorting"), with_metadata=True, copy_spike_vector=True
        )

        # load recording if possible
        if recording is None:
            # try to load the recording if not provided
            for type in ("json", "pickle"):
                filename = folder / f"recording.{type}"
                if filename.exists():
                    try:
                        recording = load(filename, base_folder=folder)
                        break
                    except:
                        recording = None
        else:
            # TODO maybe maybe not??? : do we need to check  attributes match internal rec_attributes
            # Note this will make the loading too slow
            pass

        # recording attributes
        rec_attributes_file = folder / "recording_info" / "recording_attributes.json"
        if not rec_attributes_file.exists():
            raise ValueError("This folder is not a SortingAnalyzer with format='binary_folder'")
        with open(rec_attributes_file, "r") as f:
            rec_attributes = json.load(f)
        # the probe is handle ouside the main json
        probegroup_file = folder / "recording_info" / "probegroup.json"

        if probegroup_file.is_file():
            rec_attributes["probegroup"] = probeinterface.read_probeinterface(probegroup_file)
        else:
            rec_attributes["probegroup"] = None

        # sparsity
        sparsity_file = folder / "sparsity_mask.npy"
        if sparsity_file.is_file():
            sparsity_mask = np.load(sparsity_file)
            sparsity = ChannelSparsity(sparsity_mask, sorting.unit_ids, rec_attributes["channel_ids"])
        else:
            sparsity = None

        # PATCH: Because SortingAnalyzer added this json during the development of 0.101.0 we need to save
        # this as a bridge for early adopters. The else branch can be removed in version 0.102.0/0.103.0
        # so that this can be simplified in the future
        # See https://github.com/SpikeInterface/spikeinterface/issues/2788

        settings_file = folder / f"settings.json"
        if settings_file.exists():
            with open(settings_file, "r") as f:
                settings = json.load(f)
        else:
            warnings.warn("settings.json not found for this folder writing one with return_in_uV=True")
            settings = dict(return_in_uV=True)
            with open(settings_file, "w") as f:
                json.dump(check_json(settings), f, indent=4)

        return_in_uV = settings.get("return_in_uV", settings.get("return_scaled", True))

        sorting_analyzer = SortingAnalyzer(
            sorting=sorting,
            recording=recording,
            rec_attributes=rec_attributes,
            format="binary_folder",
            sparsity=sparsity,
            return_in_uV=return_in_uV,
            backend_options=backend_options,
        )
        sorting_analyzer.folder = folder

        return sorting_analyzer

    def _get_zarr_root(self, mode="r+"):
        assert mode in ("r+", "a", "r"), "mode must be 'r+', 'a' or 'r'"

        storage_options = self._backend_options.get("storage_options", {})
        zarr_root = super_zarr_open(self.folder, mode=mode, storage_options=storage_options)
        return zarr_root

    @classmethod
    def create_zarr(cls, folder, sorting, recording, sparsity, return_in_uV, rec_attributes, backend_options):
        # used by create and save_as
        import zarr
        import numcodecs
        from .zarrextractors import add_sorting_to_zarr_group

        if is_path_remote(folder):
            remote = True
        else:
            remote = False
        if not remote:
            folder = clean_zarr_folder_name(folder)
            if folder.is_dir():
                raise ValueError(f"Folder already exists {folder}")

        backend_options = {} if backend_options is None else backend_options
        storage_options = backend_options.get("storage_options", {})
        saving_options = backend_options.get("saving_options", {})

        zarr_root = zarr.open(folder, mode="w", storage_options=storage_options)

        info = dict(version=spikeinterface.__version__, dev_mode=spikeinterface.DEV_MODE, object="SortingAnalyzer")
        zarr_root.attrs["spikeinterface_info"] = check_json(info)

        settings = dict(return_in_uV=return_in_uV)
        zarr_root.attrs["settings"] = check_json(settings)

        # the recording
        relative_to = folder if not remote else None
        if recording is not None:
            rec_dict = recording.to_dict(relative_to=relative_to, recursive=True)
            if recording.check_serializability("json"):
                # zarr_root.create_dataset("recording", data=rec_dict, object_codec=numcodecs.JSON())
                zarr_rec = np.array([check_json(rec_dict)], dtype=object)
                zarr_root.create_dataset("recording", data=zarr_rec, object_codec=numcodecs.JSON())
            elif recording.check_serializability("pickle"):
                # zarr_root.create_dataset("recording", data=rec_dict, object_codec=numcodecs.Pickle())
                zarr_rec = np.array([rec_dict], dtype=object)
                zarr_root.create_dataset("recording", data=zarr_rec, object_codec=numcodecs.Pickle())
            else:
                warnings.warn("The Recording is not serializable! The recording link will be lost for future load")
        else:
            assert rec_attributes is not None, "recording or rec_attributes must be provided"
            warnings.warn("Recording not provided, instntiating SortingAnalyzer in recordingless mode.")

        # sorting provenance
        sort_dict = sorting.to_dict(relative_to=relative_to, recursive=True)
        if sorting.check_serializability("json"):
            zarr_sort = np.array([check_json(sort_dict)], dtype=object)
            zarr_root.create_dataset("sorting_provenance", data=zarr_sort, object_codec=numcodecs.JSON())
        elif sorting.check_serializability("pickle"):
            zarr_sort = np.array([sort_dict], dtype=object)
            zarr_root.create_dataset("sorting_provenance", data=zarr_sort, object_codec=numcodecs.Pickle())
        else:
            warnings.warn(
                "The sorting provenance is not serializable! The sorting provenance link will be lost for future load"
            )

        recording_info = zarr_root.create_group("recording_info")

        if rec_attributes is None:
            rec_attributes = get_rec_attributes(recording)
            probegroup = recording.get_probegroup()
        else:
            rec_attributes = rec_attributes.copy()
            probegroup = rec_attributes.pop("probegroup")

        recording_info.attrs["recording_attributes"] = check_json(rec_attributes)

        if probegroup is not None:
            recording_info.attrs["probegroup"] = check_json(probegroup.to_dict())

        if sparsity is not None:
            zarr_root.create_dataset("sparsity_mask", data=sparsity.mask, **saving_options)

        add_sorting_to_zarr_group(sorting, zarr_root.create_group("sorting"), **saving_options)

        recording_info = zarr_root.create_group("extensions")

        zarr.consolidate_metadata(zarr_root.store)

        return cls.load_from_zarr(folder, recording=recording, backend_options=backend_options)

    @classmethod
    def load_from_zarr(cls, folder, recording=None, backend_options=None):
        import zarr
        from .loading import load

        backend_options = {} if backend_options is None else backend_options
        storage_options = backend_options.get("storage_options", {})

        zarr_root = super_zarr_open(str(folder), mode="r", storage_options=storage_options)

        si_info = zarr_root.attrs["spikeinterface_info"]
        if parse(si_info["version"]) < parse("0.101.1"):
            # v0.101.0 did not have a consolidate metadata step after computing extensions.
            # Here we try to consolidate the metadata and throw a warning if it fails.
            try:
                zarr_root_a = zarr.open(str(folder), mode="a", storage_options=storage_options)
                zarr.consolidate_metadata(zarr_root_a.store)
            except Exception as e:
                warnings.warn(
                    "The zarr store was not properly consolidated prior to v0.101.1. "
                    "This may lead to unexpected behavior in loading extensions. "
                    "Please consider re-generating the SortingAnalyzer object."
                )

        # load internal sorting in memory
        sorting = NumpySorting.from_sorting(
            ZarrSortingExtractor(folder, zarr_group="sorting", storage_options=storage_options),
            with_metadata=True,
            copy_spike_vector=True,
        )

        # load recording if possible
        if recording is None:
            rec_field = zarr_root.get("recording")
            if rec_field is not None:
                rec_dict = rec_field[0]
                try:
                    recording = load(rec_dict, base_folder=folder)
                except:
                    recording = None
        else:
            # TODO maybe maybe not??? : do we need to check  attributes match internal rec_attributes
            # Note this will make the loading too slow
            pass

        # recording attributes
        rec_attributes = zarr_root["recording_info"].attrs["recording_attributes"]
        if "probegroup" in zarr_root["recording_info"].attrs:
            probegroup_dict = zarr_root["recording_info"].attrs["probegroup"]
            rec_attributes["probegroup"] = probeinterface.ProbeGroup.from_dict(probegroup_dict)
        else:
            rec_attributes["probegroup"] = None

        # sparsity
        if "sparsity_mask" in zarr_root:
            sparsity = ChannelSparsity(
                np.array(zarr_root["sparsity_mask"]), sorting.unit_ids, rec_attributes["channel_ids"]
            )
        else:
            sparsity = None

        return_in_uV = zarr_root.attrs["settings"].get(
            "return_in_uV", zarr_root.attrs["settings"].get("return_scaled", True)
        )

        sorting_analyzer = SortingAnalyzer(
            sorting=sorting,
            recording=recording,
            rec_attributes=rec_attributes,
            format="zarr",
            sparsity=sparsity,
            return_in_uV=return_in_uV,
            backend_options=backend_options,
        )
        sorting_analyzer.folder = folder

        return sorting_analyzer

    def set_temporary_recording(self, recording: BaseRecording, check_dtype: bool = True):
        """
        Sets a temporary recording object. This function can be useful to temporarily set
        a "cached" recording object that is not saved in the SortingAnalyzer object to speed up
        computations. Upon reloading, the SortingAnalyzer object will try to reload the recording
        from the original location in a lazy way.


        Parameters
        ----------
        recording : BaseRecording
            The recording object to set as temporary recording.
        check_dtype : bool, default: True
            If True, check that the dtype of the temporary recording is the same as the original recording.
        """
        # check that recording is compatible
        attributes_match, exception_str = do_recording_attributes_match(
            recording, self.rec_attributes, check_dtype=check_dtype
        )
        if not attributes_match:
            raise ValueError(exception_str)
        if not np.array_equal(recording.get_channel_locations(), self.get_channel_locations()):
            raise ValueError("Recording channel locations do not match.")
        if self._recording is not None:
            warnings.warn("SortingAnalyzer recording is already set. The current recording is temporarily replaced.")
        self._temporary_recording = recording

    def set_sorting_property(
        self,
        key,
        values: list | np.ndarray | tuple,
        ids: list | np.ndarray | tuple | None = None,
        missing_value: Any = None,
        save: bool = True,
    ) -> None:
        """
        Set property vector for unit ids.

        If the SortingAnalyzer backend is in memory, the property will be only set in memory.
        If the SortingAnalyzer backend is `binary_folder` or `zarr`, the property will also
        be saved to to the backend.

        Parameters
        ----------
        key : str
            The property name
        values : np.array
            Array of values for the property
        ids : list/np.array, default: None
            List of subset of ids to set the values.
            if None all the ids are set or changed
        missing_value : Any, default: None
            In case the property is set on a subset of values ("ids" not None),
            This argument specifies how to fill missing values.
            The `missing_value` is required for types int and unsigned int.
        save : bool, default: True
            If True, the property is saved to the backend if possible.
        """
        self.sorting.set_property(key, values, ids=ids, missing_value=missing_value)
        if not self.is_read_only() and save:
            if self.format == "binary_folder":
                np.save(self.folder / "sorting" / "properties" / f"{key}.npy", self.sorting.get_property(key))
            elif self.format == "zarr":
                import zarr

                zarr_root = self._get_zarr_root(mode="r+")
                prop_values = self.sorting.get_property(key)
                if prop_values.dtype.kind == "O":
                    warnings.warn(f"Property {key} not saved because it is a python Object type")
                else:
                    if key in zarr_root["sorting"]["properties"]:
                        zarr_root["sorting"]["properties"][key][:] = prop_values
                    else:
                        zarr_root["sorting"]["properties"].create_dataset(name=key, data=prop_values, compressor=None)
                    # IMPORTANT: we need to re-consolidate the zarr store!
                    zarr.consolidate_metadata(zarr_root.store)

    def get_sorting_property(self, key: str, ids: Optional[Iterable] = None) -> np.ndarray:
        """
        Get property vector for unit ids.

        Parameters
        ----------
        key : str
            The property name
        ids : list/np.array, default: None
            List of subset of ids to get the values.
            if None all the ids are returned

        Returns
        -------
        values : np.array
            Array of values for the property
        """
        return self.sorting.get_property(key, ids=ids)

    def are_units_mergeable(
        self,
        merge_unit_groups: list[str | int],
        merging_mode: str = "soft",
        sparsity_overlap: float = 0.75,
        return_masks: bool = False,
    ):
        """
        Check if soft merges can be performed given sparsity_overlap param.

        Parameters
        ----------
        merge_unit_groups : list/tuple of lists/tuples
            A list of lists for every merge group. Each element needs to have at least two elements
            (two units to merge).
        merging_mode : "soft" | "hard", default: "soft"
            How merges are performed. In the "soft" mode, merges will be approximated, with no smart merging
            of the extension data.
        sparsity_overlap : float, default: 0.75
            The percentage of overlap that units should share in order to accept merges.
        return_masks : bool, default: False
            If True, return the masks used for the merge.

        Returns
        -------
        mergeable : dict[bool]
            Dictionary of of mergeable units. The keys are the merge unit groups (as tuple), and boolean
            values indicate if the merge is possible.
        masks : dict[np.array]
            Dictionary of masks used for the merge. The keys are the merge unit groups, and the values
            are the masks used for the merge.
        """
        mergeable = {}
        masks = {}

        for merge_unit_group in merge_unit_groups:
            merge_unit_indices = self.sorting.ids_to_indices(merge_unit_group)
            union_mask = np.sum(self.sparsity.mask[merge_unit_indices], axis=0) > 0
            intersection_mask = np.prod(self.sparsity.mask[merge_unit_indices], axis=0) > 0
            thr = np.sum(intersection_mask) / np.sum(union_mask)

            if self.sparsity is None or merging_mode == "hard":
                mergeable[tuple(merge_unit_group)] = True
                masks[tuple(merge_unit_group)] = union_mask
            else:
                mergeable[tuple(merge_unit_group)] = thr >= sparsity_overlap
                masks[tuple(merge_unit_group)] = intersection_mask

        if return_masks:
            return mergeable, masks
        else:
            return mergeable

    def _save_or_select_or_merge_or_split(
        self,
        format="binary_folder",
        folder=None,
        unit_ids=None,
        merge_unit_groups=None,
        censor_ms=None,
        merging_mode="soft",
        sparsity_overlap=0.75,
        merge_new_unit_ids=None,
        split_units=None,
        splitting_mode="soft",
        split_new_unit_ids=None,
        backend_options=None,
        verbose=False,
        **job_kwargs,
    ) -> "SortingAnalyzer":
        """
        Internal method used by both `save_as()`, `copy()`, `select_units()`, and `merge_units()`.

        Parameters
        ----------
        format : "memory" | "binary_folder" | "zarr", default: "binary_folder"
            The format to save the SortingAnalyzer object
        folder : str | Path | None, default: None
            The folder where the SortingAnalyzer object will be saved
        unit_ids : list or None, default: None
            The unit ids to keep in the new SortingAnalyzer object. If `merge_unit_groups` is not None,
            `unit_ids` must be given it must contain all unit_ids.
        merge_unit_groups : list/tuple of lists/tuples or None, default: None
            A list of lists for every merge group. Each element needs to have at least two elements
            (two units to merge). If `merge_unit_groups` is not None, `new_unit_ids` must be given.
        censor_ms : None or float, default: None
            When merging units, any spikes violating this refractory period will be discarded.
        merging_mode : "soft" | "hard", default: "soft"
            How merges are performed. In the "soft" mode, merges will be approximated, with no smart merging
            of the extension data. In the "hard" mode, the extensions for merged units will be recomputed.
        sparsity_overlap : float, default 0.75
            The percentage of overlap that units should share in order to accept merges. If this criteria is not
            achieved, soft merging will not be performed.
        merge_new_unit_ids : list or None, default: None
            The new unit ids for merged units. Required if `merge_unit_groups` is not None.
        split_units : dict or None, default: None
            A dictionary with the keys being the unit ids to split and the values being the split indices.
        splitting_mode : "soft" | "hard", default: "soft"
            How splits are performed. In the "soft" mode, splits will be approximated, with no smart splitting.
            If `splitting_mode` is "hard", the extensons for split units willbe recomputed.
        split_new_unit_ids : list or None, default: None
            The new unit ids for split units. Required if `split_units` is not None.
        verbose : bool, default: False
            If True, output is verbose.
        backend_options : dict | None, default: None
            Keyword arguments for the backend specified by format. It can contain the:

                * storage_options: dict | None (fsspec storage options)
                * saving_options: dict | None (additional saving options for creating and saving datasets, e.g. compression/filters for zarr)
        job_kwargs : keyword arguments
            Keyword arguments for the job parallelization.

        Returns
        -------
        new_sorting_analyzer : SortingAnalyzer
            The newly created SortingAnalyzer object.
        """
        if self.has_recording():
            recording = self._recording
        elif self.has_temporary_recording():
            recording = self._temporary_recording
        else:
            recording = None

        has_removed = unit_ids is not None
        has_merges = merge_unit_groups is not None
        has_splits = split_units is not None
        assert not has_merges if has_splits else True, "Cannot merge and split at the same time"

        if self.sparsity is not None:
            if not has_removed and not has_merges and not has_splits:
                # no changes in units
                sparsity = self.sparsity
            elif has_removed and not has_merges and not has_splits:
                # remove units
                sparsity_mask = self.sparsity.mask[np.isin(self.unit_ids, unit_ids), :]
                sparsity = ChannelSparsity(sparsity_mask, unit_ids, self.channel_ids)
            elif has_merges:
                # merge units
                all_unit_ids = unit_ids
                sparsity_mask = np.zeros((len(all_unit_ids), self.sparsity.mask.shape[1]), dtype=bool)
                mergeable, masks = self.are_units_mergeable(
                    merge_unit_groups,
                    sparsity_overlap=sparsity_overlap,
                    merging_mode=merging_mode,
                    return_masks=True,
                )

                for unit_index, unit_id in enumerate(all_unit_ids):
                    if unit_id in merge_new_unit_ids:
                        merge_unit_group = tuple(merge_unit_groups[merge_new_unit_ids.index(unit_id)])
                        if not mergeable[merge_unit_group]:
                            raise Exception(
                                f"The sparsity of {merge_unit_group} do not overlap enough for a soft merge using "
                                f"a sparsity threshold of {sparsity_overlap}. You can either lower the threshold or use "
                                "a hard merge."
                            )
                        else:
                            sparsity_mask[unit_index] = masks[merge_unit_group]
                    else:
                        # This means that the unit is already in the previous sorting
                        index = self.sorting.id_to_index(unit_id)
                        sparsity_mask[unit_index] = self.sparsity.mask[index]
                sparsity = ChannelSparsity(sparsity_mask, list(all_unit_ids), self.channel_ids)
            elif has_splits:
                # split units
                all_unit_ids = unit_ids
                original_unit_ids = self.unit_ids
                sparsity_mask = np.zeros((len(all_unit_ids), self.sparsity.mask.shape[1]), dtype=bool)
                for unit_index, unit_id in enumerate(all_unit_ids):
                    if unit_id not in original_unit_ids:
                        # then it is a new unit
                        # we assign the original sparsity
                        for split_unit, new_unit_ids in zip(split_units, split_new_unit_ids):
                            if unit_id in new_unit_ids:
                                original_unit_index = self.sorting.id_to_index(split_unit)
                                sparsity_mask[unit_index] = self.sparsity.mask[original_unit_index]
                                break
                    else:
                        original_unit_index = self.sorting.id_to_index(unit_id)
                        sparsity_mask[unit_index] = self.sparsity.mask[original_unit_index]
                sparsity = ChannelSparsity(sparsity_mask, list(all_unit_ids), self.channel_ids)
        else:
            sparsity = None

        # Note that the sorting is a copy we need to go back to the orginal sorting (if available)
        sorting_provenance = self.get_sorting_provenance()
        if sorting_provenance is None:
            # if the original sorting object is not available anymore (kilosort folder deleted, ....), take the copy
            sorting_provenance = self.sorting

        if merge_unit_groups is None and split_units is None:
            # when only some unit_ids then the sorting must be sliced
            # TODO check that unit_ids are in same order otherwise many extension do handle it properly!!!!
            sorting_provenance = sorting_provenance.select_units(unit_ids)
        elif merge_unit_groups is not None:
            assert split_units is None, "split_units must be None when merge_unit_groups is None"
            from spikeinterface.core.sorting_tools import apply_merges_to_sorting

            sorting_provenance, keep_mask, _ = apply_merges_to_sorting(
                sorting=sorting_provenance,
                merge_unit_groups=merge_unit_groups,
                new_unit_ids=merge_new_unit_ids,
                censor_ms=censor_ms,
                return_extra=True,
            )
            if censor_ms is None:
                # in this case having keep_mask None is faster instead of having a vector of ones
                keep_mask = None
        elif split_units is not None:
            assert merge_unit_groups is None, "merge_unit_groups must be None when split_units is not None"
            from spikeinterface.core.sorting_tools import apply_splits_to_sorting

            sorting_provenance = apply_splits_to_sorting(
                sorting=sorting_provenance,
                unit_splits=split_units,
                new_unit_ids=split_new_unit_ids,
            )

        backend_options = {} if backend_options is None else backend_options

        if format == "memory":
            # This make a copy of actual SortingAnalyzer
            new_sorting_analyzer = SortingAnalyzer.create_memory(
                sorting_provenance, recording, sparsity, self.return_in_uV, self.rec_attributes
            )

        elif format == "binary_folder":
            # create  a new folder
            assert folder is not None, "For format='binary_folder' folder must be provided"
            folder = Path(folder)
            new_sorting_analyzer = SortingAnalyzer.create_binary_folder(
                folder,
                sorting_provenance,
                recording,
                sparsity,
                self.return_in_uV,
                self.rec_attributes,
                backend_options=backend_options,
            )

        elif format == "zarr":
            assert folder is not None, "For format='zarr' folder must be provided"
            folder = clean_zarr_folder_name(folder)
            new_sorting_analyzer = SortingAnalyzer.create_zarr(
                folder,
                sorting_provenance,
                recording,
                sparsity,
                self.return_in_uV,
                self.rec_attributes,
                backend_options=backend_options,
            )
        else:
            raise ValueError(f"SortingAnalyzer.save: unsupported format: {format}")

        # make a copy of extensions
        # note that the copy of extension handle itself the slicing of units when necessary and also the saveing
        sorted_extensions = _sort_extensions_by_dependency(self.extensions)
        # hack: quality metrics are computed at last
        qm_extension_params = sorted_extensions.pop("quality_metrics", None)
        if qm_extension_params is not None:
            sorted_extensions["quality_metrics"] = qm_extension_params
        recompute_dict = {}

        for extension_name, extension in sorted_extensions.items():
            if merge_unit_groups is None and split_units is None:
                # copy full or select
                new_sorting_analyzer.extensions[extension_name] = extension.copy(
                    new_sorting_analyzer, unit_ids=unit_ids
                )
            elif merge_unit_groups is not None:
                # merge
                if merging_mode == "soft":
                    new_sorting_analyzer.extensions[extension_name] = extension.merge(
                        new_sorting_analyzer,
                        merge_unit_groups=merge_unit_groups,
                        new_unit_ids=merge_new_unit_ids,
                        keep_mask=keep_mask,
                        verbose=verbose,
                        **job_kwargs,
                    )
                elif merging_mode == "hard":
                    recompute_dict[extension_name] = extension.params
            else:
                # split
                if splitting_mode == "soft":
                    new_sorting_analyzer.extensions[extension_name] = extension.split(
                        new_sorting_analyzer, split_units=split_units, new_unit_ids=split_new_unit_ids, verbose=verbose
                    )
                elif splitting_mode == "hard":
                    recompute_dict[extension_name] = extension.params

        if len(recompute_dict) > 0:
            new_sorting_analyzer.compute_several_extensions(recompute_dict, save=True, verbose=verbose, **job_kwargs)

        return new_sorting_analyzer

    def save_as(self, format="memory", folder=None, backend_options=None) -> "SortingAnalyzer":
        """
        Save SortingAnalyzer object into another format.
        Uselful for memory to zarr or memory to binary.

        Note that the recording provenance or sorting provenance can be lost.

        Mainly propagates the copied sorting and recording properties.

        Parameters
        ----------
        folder : str | Path | None, default: None
            The output folder if `format` is "zarr" or "binary_folder"
        format : "memory" | "binary_folder" | "zarr", default: "memory"
            The new backend format to use
        backend_options : dict | None, default: None
            Keyword arguments for the backend specified by format. It can contain the:

                * storage_options: dict | None (fsspec storage options)
                * saving_options: dict | None (additional saving options for creating and saving datasets, e.g. compression/filters for zarr)
        """
        if format == "zarr":
            folder = clean_zarr_folder_name(folder)
        return self._save_or_select_or_merge_or_split(format=format, folder=folder, backend_options=backend_options)

    def select_units(self, unit_ids, format="memory", folder=None) -> "SortingAnalyzer":
        """
        This method is equivalent to `save_as()` but with a subset of units.
        Filters units by creating a new sorting analyzer object in a new folder.

        Extensions are also updated to filter the selected unit ids.

        Parameters
        ----------
        unit_ids : list or array
            The unit ids to keep in the new SortingAnalyzer object
        format : "memory" | "binary_folder" | "zarr" , default: "memory"
            The format of the returned SortingAnalyzer.
        folder : Path | None, deafult: None
            The new folder where the analyzer with selected units is copied if `format` is
            "binary_folder" or "zarr"

        Returns
        -------
        analyzer :  SortingAnalyzer
            The newly create sorting_analyzer with the selected units
        """
        # TODO check that unit_ids are in same order otherwise many extension do handle it properly!!!!
        if format == "zarr":
            folder = clean_zarr_folder_name(folder)
        return self._save_or_select_or_merge_or_split(format=format, folder=folder, unit_ids=unit_ids)

    def remove_units(self, remove_unit_ids, format="memory", folder=None) -> "SortingAnalyzer":
        """
        This method is equivalent to `save_as()` but with removal of a subset of units.
        Filters units by creating a new sorting analyzer object in a new folder.

        Extensions are also updated to remove the unit ids.

        Parameters
        ----------
        remove_unit_ids : list or array
            The unit ids to remove in the new SortingAnalyzer object.
        format : "memory" | "binary_folder" | "zarr" , default: "memory"
            The format of the returned SortingAnalyzer.
        folder : Path or None, default: None
            The new folder where the analyzer without removed units is copied if `format`
            is "binary_folder" or "zarr"

        Returns
        -------
        analyzer :  SortingAnalyzer
            The newly create sorting_analyzer with the selected units
        """
        # TODO check that unit_ids are in same order otherwise many extension do handle it properly!!!!
        unit_ids = self.unit_ids[~np.isin(self.unit_ids, remove_unit_ids)]
        if format == "zarr":
            folder = clean_zarr_folder_name(folder)
        return self._save_or_select_or_merge_or_split(format=format, folder=folder, unit_ids=unit_ids)

    def merge_units(
        self,
        merge_unit_groups: list[list[str | int]] | list[tuple[str | int]],
        new_unit_ids: list[int | str] | None = None,
        censor_ms: float | None = None,
        merging_mode: str = "soft",
        sparsity_overlap: float = 0.75,
        new_id_strategy: str = "append",
        return_new_unit_ids: bool = False,
        format: str = "memory",
        folder: Path | str | None = None,
        verbose: bool = False,
        **job_kwargs,
    ) -> "SortingAnalyzer | tuple[SortingAnalyzer, list[int | str]]":
        """
        This method is equivalent to `save_as()` but with a list of merges that have to be achieved.
        Merges units by creating a new SortingAnalyzer object with the appropriate merges

        Extensions are also updated to display the merged `unit_ids`.

        Parameters
        ----------
        merge_unit_groups : list/tuple of lists/tuples
            A list of lists for every merge group. Each element needs to have at least two elements (two units to merge),
            but it can also have more (merge multiple units at once).
        new_unit_ids : None | list, default: None
            A new unit_ids for merged units. If given, it needs to have the same length as `merge_unit_groups`. If None,
            merged units will have the first unit_id of every lists of merges
        censor_ms : None | float, default: None
            When merging units, any spikes violating this refractory period will be discarded. If None all units are kept
        merging_mode : ["soft", "hard"], default: "soft"
            How merges are performed. If the `merge_mode` is "soft" , merges will be approximated, with no reloading of the
            waveforms. This will lead to approximations. If `merge_mode` is "hard", recomputations are accurately performed,
            reloading waveforms if needed
        sparsity_overlap : float, default 0.75
            The percentage of overlap that units should share in order to accept merges. If this criteria is not
            achieved, soft merging will not be possible and an error will be raised
        new_id_strategy : "append" | "take_first", default: "append"
            The strategy that should be used, if `new_unit_ids` is None, to create new unit_ids.

                * "append" : new_units_ids will be added at the end of max(sorting.unit_ids)
                * "take_first" : new_unit_ids will be the first unit_id of every list of merges
        return_new_unit_ids : bool, default False
            Alse return new_unit_ids which are the ids of the new units.
        folder : Path | None, default: None
            The new folder where the analyzer with merged units is copied for `format` "binary_folder" or "zarr"
        format : "memory" | "binary_folder" | "zarr", default: "memory"
            The format of SortingAnalyzer
        verbose : bool, default: False
            Whether to display calculations (such as sparsity estimation)

        Returns
        -------
        analyzer :  SortingAnalyzer
            The newly create `SortingAnalyzer` with the selected units
        """

        if format == "zarr":
            folder = clean_zarr_folder_name(folder)

        assert merging_mode in ["soft", "hard"], "Merging mode should be either soft or hard"

        if len(merge_unit_groups) == 0:
            raise ValueError("Merging requires at least one group of units to merge")

        for units in merge_unit_groups:
            if len(units) < 2:
                raise ValueError("Merging requires at least two units to merge")

        new_unit_ids = generate_unit_ids_for_merge_group(
            self.unit_ids, merge_unit_groups, new_unit_ids, new_id_strategy
        )
        all_unit_ids = _get_ids_after_merging(self.unit_ids, merge_unit_groups, new_unit_ids=new_unit_ids)

        new_analyzer = self._save_or_select_or_merge_or_split(
            format=format,
            folder=folder,
            merge_unit_groups=merge_unit_groups,
            unit_ids=all_unit_ids,
            censor_ms=censor_ms,
            merging_mode=merging_mode,
            sparsity_overlap=sparsity_overlap,
            verbose=verbose,
            merge_new_unit_ids=new_unit_ids,
            **job_kwargs,
        )
        if return_new_unit_ids:
            return new_analyzer, new_unit_ids
        else:
            return new_analyzer

    def split_units(
        self,
        split_units: dict[list[str | int], list[int] | list[list[int]]],
        new_unit_ids: list[list[int | str]] | None = None,
        new_id_strategy: str = "append",
        return_new_unit_ids: bool = False,
        format: str = "memory",
        folder: Path | str | None = None,
        verbose: bool = False,
        **job_kwargs,
    ) -> "SortingAnalyzer | tuple[SortingAnalyzer, list[int | str]]":
        """
        This method is equivalent to `save_as()` but with a list of splits that have to be achieved.
        Split units by creating a new SortingAnalyzer object with the appropriate splits

        Extensions are also updated to display the split `unit_ids`.

        Parameters
        ----------
        split_units : dict
            A dictionary with the keys being the unit ids to split and the values being the split indices.
            The split indices for each unit MUST be a list of lists, where each sublist (at least two) contains the
            indices of the spikes to be assigned to the each split. The sum of the lengths of the sublists must equal
            the number of spikes in the unit.
        new_unit_ids : None | list, default: None
            A new unit_ids for split units. If given, it needs to have the same length as `merge_unit_groups`. If None,
            merged units will have the first unit_id of every lists of merges
        new_id_strategy : "append" | "split", default: "append"
            The strategy that should be used, if `new_unit_ids` is None, to create new unit_ids.

                * "append" : new_units_ids will be added at the end of max(sorting.unit_ids)
                * "split" : new_unit_ids will be the original unit_id to split with -{subsplit}
        return_new_unit_ids : bool, default False
            Alse return new_unit_ids which are the ids of the new units.
        folder : Path | None, default: None
            The new folder where the analyzer with merged units is copied for `format` "binary_folder" or "zarr"
        format : "memory" | "binary_folder" | "zarr", default: "memory"
            The format of SortingAnalyzer
        verbose : bool, default: False
            Whether to display calculations (such as sparsity estimation)

        Returns
        -------
        analyzer :  SortingAnalyzer
            The newly create `SortingAnalyzer` with the selected units
        """

        if format == "zarr":
            folder = clean_zarr_folder_name(folder)

        if len(split_units) == 0:
            raise ValueError("Splitting requires at least one unit to split")

        check_unit_splits_consistency(split_units, self.sorting)

        new_unit_ids = generate_unit_ids_for_split(self.unit_ids, split_units, new_unit_ids, new_id_strategy)
        all_unit_ids = _get_ids_after_splitting(self.unit_ids, split_units, new_unit_ids=new_unit_ids)

        new_analyzer = self._save_or_select_or_merge_or_split(
            format=format,
            folder=folder,
            split_units=split_units,
            unit_ids=all_unit_ids,
            verbose=verbose,
            split_new_unit_ids=new_unit_ids,
            **job_kwargs,
        )
        if return_new_unit_ids:
            return new_analyzer, new_unit_ids
        else:
            return new_analyzer

    def copy(self):
        """
        Create a a copy of SortingAnalyzer with format "memory".
        """
        return self._save_or_select_or_merge_or_split(format="memory", folder=None)

    def is_read_only(self) -> bool:
        if self.format == "memory":
            return False
        elif self.format == "binary_folder":
            return not os.access(self.folder, os.W_OK)
        else:
            if not is_path_remote(str(self.folder)):
                return not os.access(self.folder, os.W_OK)
            else:
                # in this case we don't know if the file is read only so an error
                # will be raised if we try to save/append
                return False

    ## map attribute and property zone

    @property
    def recording(self) -> BaseRecording:
        if not self.has_recording() and not self.has_temporary_recording():
            raise ValueError("SortingAnalyzer could not load the recording")
        return self._temporary_recording or self._recording

    @property
    def channel_ids(self) -> np.ndarray:
        return np.array(self.rec_attributes["channel_ids"])

    @property
    def sampling_frequency(self) -> float:
        return self.sorting.get_sampling_frequency()

    @property
    def unit_ids(self) -> np.ndarray:
        return self.sorting.unit_ids

    def has_recording(self) -> bool:
        return self._recording is not None

    def has_temporary_recording(self) -> bool:
        return self._temporary_recording is not None

    def is_sparse(self) -> bool:
        return self.sparsity is not None

    def is_filtered(self) -> bool:
        return self.rec_attributes["is_filtered"]

    def get_sorting_provenance(self):
        """
        Get the original sorting if possible otherwise return None
        """
        from .loading import load

        if self.format == "memory":
            # the orginal sorting provenance is not keps in that case
            sorting_provenance = None

        elif self.format == "binary_folder":
            for type in ("json", "pickle"):
                filename = self.folder / f"sorting_provenance.{type}"
                sorting_provenance = None
                if filename.exists():
                    # try-except here is because it's not required to be able
                    # to load the sorting provenance, as the user might have deleted
                    # the original sorting folder
                    try:
                        sorting_provenance = load(filename, base_folder=self.folder)
                        break
                    except:
                        pass
                        # sorting_provenance = None

        elif self.format == "zarr":
            zarr_root = self._get_zarr_root(mode="r")
            sorting_provenance = None
            if "sorting_provenance" in zarr_root.keys():
                # try-except here is because it's not required to be able
                # to load the sorting provenance, as the user might have deleted
                # the original sorting folder
                try:
                    sort_dict = zarr_root["sorting_provenance"][0]
                    sorting_provenance = load(sort_dict, base_folder=self.folder)
                except:
                    pass

        return sorting_provenance

    def get_num_samples(self, segment_index: Optional[int] = None) -> int:
        # we use self.sorting to check segment_index
        segment_index = self.sorting._check_segment_index(segment_index)
        return self.rec_attributes["num_samples"][segment_index]

    def get_total_samples(self) -> int:
        s = 0
        for segment_index in range(self.get_num_segments()):
            s += self.get_num_samples(segment_index)
        return s

    def get_total_duration(self) -> float:
        if self.has_recording() or self.has_temporary_recording():
            duration = self.recording.get_total_duration()
        else:
            duration = self.get_total_samples() / self.sampling_frequency
        return duration

    def get_num_channels(self) -> int:
        return self.rec_attributes["num_channels"]

    def get_num_segments(self) -> int:
        return self.sorting.get_num_segments()

    def get_probegroup(self):
        return self.rec_attributes["probegroup"]

    def get_probe(self):
        probegroup = self.get_probegroup()
        assert len(probegroup.probes) == 1, "There are several probes. Use `get_probegroup()`"
        return probegroup.probes[0]

    def get_channel_locations(self) -> np.ndarray:
        # important note : contrary to recording
        # this give all channel locations, so no kwargs like channel_ids and axes
        probegroup = self.get_probegroup()
        probe_as_numpy_array = probegroup.to_numpy(complete=True)
        # we need to sort by device_channel_indices to ensure the order of locations is correct
        probe_as_numpy_array = probe_as_numpy_array[np.argsort(probe_as_numpy_array["device_channel_indices"])]
        ndim = probegroup.ndim
        locations = np.zeros((probe_as_numpy_array.size, ndim), dtype="float64")
        # here we only loop through xy because only 2d locations are supported
        for i, dim in enumerate(["x", "y"][:ndim]):
            locations[:, i] = probe_as_numpy_array[dim]
        return locations

    def channel_ids_to_indices(self, channel_ids) -> np.ndarray:
        all_channel_ids = list(self.rec_attributes["channel_ids"])
        indices = np.array([all_channel_ids.index(id) for id in channel_ids], dtype=int)
        return indices

    def get_recording_property(self, key) -> np.ndarray:
        values = np.array(self.rec_attributes["properties"].get(key, None))
        return values

    def get_sorting_property(self, key) -> np.ndarray:
        return self.sorting.get_property(key)

    def get_dtype(self):
        return self.rec_attributes["dtype"]

    def get_num_units(self) -> int:
        return self.sorting.get_num_units()

    ## extensions zone
    def compute(self, input, save=True, extension_params=None, verbose=False, **kwargs) -> "AnalyzerExtension | None":
        """
        Compute one extension or several extensiosn.
        Internally calls compute_one_extension() or compute_several_extensions() depending on the input type.

        Parameters
        ----------
        input : str or dict or list
            The extensions to compute, which can be passed as:
            * a string: compute one extension. Additional parameters can be passed as key word arguments.
            * a dict: compute several extensions. The keys are the extension names and the values are dictionaries with the extension parameters.
            * a list: compute several extensions. The list contains the extension names. Additional parameters can be passed with the extension_params
            argument.
        save : bool, default: True
            If True the extension is saved to disk (only if sorting analyzer format is not "memory")
        extension_params : dict or None, default: None
            If input is a list, this parameter can be used to specify parameters for each extension.
            The extension_params keys must be included in the input list.
        **kwargs:
            All other kwargs are transmitted to extension.set_params() (if input is a string) or job_kwargs

        Returns
        -------
        extension : SortingAnalyzerExtension | None
            The extension instance if input is a string, None otherwise.

        Examples
        --------
        This function accepts the following possible signatures for flexibility:

        Compute one extension, with parameters:
        >>> analyzer.compute("waveforms", ms_before=1.5, ms_after=2.5)

        Compute two extensions with a list as input and with default parameters:
        >>> analyzer.compute(["random_spikes", "waveforms"])

        Compute two extensions with dict as input, one dict per extension
        >>> analyzer.compute({"random_spikes":{}, "waveforms":{"ms_before":1.5, "ms_after", "2.5"}})

        Compute two extensions with an input list specifying custom parameters for one
        (the other will use default parameters):
        >>> analyzer.compute(\
["random_spikes", "waveforms"],\
extension_params={"waveforms":{"ms_before":1.5, "ms_after": "2.5"}}\
)

        """
        if isinstance(input, str):
            return self.compute_one_extension(extension_name=input, save=save, verbose=verbose, **kwargs)
        elif isinstance(input, dict):
            params_, job_kwargs = split_job_kwargs(kwargs)
            assert len(params_) == 0, "Too many arguments for SortingAnalyzer.compute_several_extensions()"
            self.compute_several_extensions(extensions=input, save=save, verbose=verbose, **job_kwargs)
        elif isinstance(input, list):
            params_, job_kwargs = split_job_kwargs(kwargs)
            assert len(params_) == 0, "Too many arguments for SortingAnalyzer.compute_several_extensions()"
            extensions = {k: {} for k in input}
            if extension_params is not None:
                for ext_name, ext_params in extension_params.items():
                    assert (
                        ext_name in input
                    ), f"SortingAnalyzer.compute(): Parameters specified for {ext_name}, which is not in the specified {input}"
                    extensions[ext_name] = ext_params
            self.compute_several_extensions(extensions=extensions, save=save, verbose=verbose, **job_kwargs)
        else:
            raise ValueError("SortingAnalyzer.compute() needs a str, dict or list")

    def compute_one_extension(self, extension_name, save=True, verbose=False, **kwargs) -> "AnalyzerExtension":
        """
        Compute one extension.

        Important note: when computing again an extension, all extensions that depend on it
        will be automatically and silently deleted to keep a coherent data.

        Parameters
        ----------
        extension_name : str
            The name of the extension.
            For instance "waveforms", "templates", ...
        save : bool, default: True
            It the extension can be saved then it is saved.
            If not then the extension will only live in memory as long as the object is deleted.
            save=False is convenient to try some parameters without changing an already saved extension.

        **kwargs:
            All other kwargs are transmitted to extension.set_params() or job_kwargs

        Returns
        -------
        result_extension : AnalyzerExtension
            Return the extension instance

        Examples
        --------

        >>> Note that the return is the instance extension.
        >>> extension = sorting_analyzer.compute("waveforms", **some_params)
        >>> extension = sorting_analyzer.compute_one_extension("waveforms", **some_params)
        >>> wfs = extension.data["waveforms"]
        >>> # Note this can be be done in the old way style BUT the return is not the same it return directly data
        >>> wfs = compute_waveforms(sorting_analyzer, **some_params)

        """
        extension_class = get_extension_class(extension_name)

        for child in _get_children_dependencies(extension_name):
            if self.has_extension(child):
                print(f"Deleting {child}")
                self.delete_extension(child)

        params, job_kwargs = split_job_kwargs(kwargs)

        # check dependencies
        if extension_class.need_recording:
            assert (
                self.has_recording() or self.has_temporary_recording()
            ), f"Extension {extension_name} requires the recording"
        for dependency_name in extension_class.depend_on:
            if "|" in dependency_name:
                ok = any(self.get_extension(name) is not None for name in dependency_name.split("|"))
            else:
                ok = self.get_extension(dependency_name) is not None
            assert ok, f"Extension {extension_name} requires {dependency_name} to be computed first"

        extension_instance = extension_class(self)
        extension_instance.set_params(save=save, **params)
        if extension_class.need_job_kwargs:
            extension_instance.run(save=save, verbose=verbose, **job_kwargs)
        else:
            extension_instance.run(save=save, verbose=verbose)

        self.extensions[extension_name] = extension_instance
        return extension_instance

    def compute_several_extensions(self, extensions, save=True, verbose=False, **job_kwargs):
        """
        Compute several extensions

        Important note: when computing again an extension, all extensions that depend on it
        will be automatically and silently deleted to keep a coherent data.


        Parameters
        ----------
        extensions : dict
            Keys are extension_names and values are params.
        save : bool, default: True
            It the extension can be saved then it is saved.
            If not then the extension will only live in memory as long as the object is deleted.
            save=False is convenient to try some parameters without changing an already saved extension.

        Returns
        -------
        No return

        Examples
        --------

        >>> sorting_analyzer.compute({"waveforms": {"ms_before": 1.2}, "templates" : {"operators": ["average", "std", ]} })
        >>> sorting_analyzer.compute_several_extensions({"waveforms": {"ms_before": 1.2}, "templates" : {"operators": ["average", "std"]}})

        """

        sorted_extensions = _sort_extensions_by_dependency(extensions)

        for extension_name in sorted_extensions.keys():
            for child in _get_children_dependencies(extension_name):
                self.delete_extension(child)

        extensions_with_pipeline = {}
        extensions_without_pipeline = {}
        extensions_post_pipeline = {}
        for extension_name, extension_params in sorted_extensions.items():
            if extension_name == "quality_metrics":
                # PATCH: the quality metric is computed after the pipeline, since some of the metrics optionally require
                # the output of the pipeline extensions (e.g., spike_amplitudes, spike_locations).
                extensions_post_pipeline[extension_name] = extension_params
                continue
            extension_class = get_extension_class(extension_name)
            if extension_class.use_nodepipeline:
                extensions_with_pipeline[extension_name] = extension_params
            else:
                extensions_without_pipeline[extension_name] = extension_params

        # First extensions without pipeline
        for extension_name, extension_params in extensions_without_pipeline.items():
            extension_class = get_extension_class(extension_name)
            if extension_class.need_job_kwargs:
                self.compute_one_extension(extension_name, save=save, verbose=verbose, **extension_params, **job_kwargs)
            else:
                self.compute_one_extension(extension_name, save=save, verbose=verbose, **extension_params)
        # then extensions with pipeline
        if len(extensions_with_pipeline) > 0:
            all_nodes = []
            result_routage = []
            extension_instances = {}

            for extension_name, extension_params in extensions_with_pipeline.items():
                extension_class = get_extension_class(extension_name)
                assert (
                    self.has_recording() or self.has_temporary_recording()
                ), f"Extension {extension_name} requires the recording"

                for variable_name in extension_class.nodepipeline_variables:
                    result_routage.append((extension_name, variable_name))

                extension_instance = extension_class(self)
                extension_instance.set_params(save=save, **extension_params)
                extension_instances[extension_name] = extension_instance

                nodes = extension_instance.get_pipeline_nodes()
                all_nodes.extend(nodes)

            job_name = "Compute : " + " + ".join(extensions_with_pipeline.keys())

            t_start = perf_counter()
            results = run_node_pipeline(
                self.recording,
                all_nodes,
                job_kwargs=job_kwargs,
                job_name=job_name,
                gather_mode="memory",
                squeeze_output=False,
                verbose=verbose,
            )
            t_end = perf_counter()
            # for pipeline node extensions we can only track the runtime of the run_node_pipeline
            runtime_s = t_end - t_start

            for r, result in enumerate(results):
                extension_name, variable_name = result_routage[r]
                extension_instances[extension_name].data[variable_name] = result
                extension_instances[extension_name].run_info["runtime_s"] = runtime_s
                extension_instances[extension_name].run_info["run_completed"] = True

            for extension_name, extension_instance in extension_instances.items():
                self.extensions[extension_name] = extension_instance
                if save:
                    extension_instance.save()

        # PATCH: the quality metric is computed after the pipeline, since some of the metrics optionally require
        # the output of the pipeline extensions (e.g., spike_amplitudes, spike_locations).
        # An alternative could be to extend the "depend_on" attribute to use optional and to check if an extension
        # depends on the output of the pipeline nodes (e.g. depend_on=["spike_amplitudes[optional]"])
        for extension_name, extension_params in extensions_post_pipeline.items():
            extension_class = get_extension_class(extension_name)
            if extension_class.need_job_kwargs:
                self.compute_one_extension(extension_name, save=save, verbose=verbose, **extension_params, **job_kwargs)
            else:
                self.compute_one_extension(extension_name, save=save, verbose=verbose, **extension_params)

    def get_saved_extension_names(self):
        """
        Get extension names saved in folder or zarr that can be loaded.
        This do not load data, this only explores the directory.
        """
        saved_extension_names = []
        if self.format == "binary_folder":
            ext_folder = self.folder / "extensions"
            if ext_folder.is_dir():
                for extension_folder in ext_folder.iterdir():
                    is_saved = extension_folder.is_dir() and (extension_folder / "params.json").is_file()
                    if not is_saved:
                        continue
                    saved_extension_names.append(extension_folder.stem)

        elif self.format == "zarr":
            zarr_root = self._get_zarr_root(mode="r")
            if "extensions" in zarr_root.keys():
                extension_group = zarr_root["extensions"]
                for extension_name in extension_group.keys():
                    if "params" in extension_group[extension_name].attrs.keys():
                        saved_extension_names.append(extension_name)

        else:
            raise ValueError("SortingAnalyzer.get_saved_extension_names() works only with binary_folder and zarr")

        return saved_extension_names

    def get_extension(self, extension_name: str):
        """
        Get a AnalyzerExtension.
        If not loaded then load is automatic.

        Return None if the extension is not computed yet (this avoids the use of has_extension() and then get it)

        """
        if extension_name in self.extensions:
            return self.extensions[extension_name]

        elif self.format != "memory" and self.has_extension(extension_name):
            self.load_extension(extension_name)
            return self.extensions[extension_name]

        else:
            return None

    def load_extension(self, extension_name: str):
        """
        Load an extension from a folder or zarr into the `ResultSorting.extensions` dict.

        Parameters
        ----------
        extension_name : str
            The extension name.

        Returns
        -------
        ext_instance:
            The loaded instance of the extension

        """
        assert (
            self.format != "memory"
        ), "SortingAnalyzer.load_extension() does not work for format='memory' use SortingAnalyzer.get_extension() instead"

        extension_class = get_extension_class(extension_name)

        extension_instance = extension_class.load(self)

        self.extensions[extension_name] = extension_instance

        return extension_instance

    def load_all_saved_extension(self):
        """
        Load all saved extensions in memory.
        """
        for extension_name in self.get_saved_extension_names():
            self.load_extension(extension_name)

    def delete_extension(self, extension_name) -> None:
        """
        Delete the extension from the dict and also in the persistent zarr or folder.
        """

        # delete from folder or zarr
        if self.format != "memory" and self.has_extension(extension_name):
            # need a reload to reset the folder
            ext = self.load_extension(extension_name)
            ext.delete()

        # remove from dict
        self.extensions.pop(extension_name, None)

    def get_loaded_extension_names(self):
        """
        Return the loaded or already computed extensions names.
        """
        return list(self.extensions.keys())

    def has_extension(self, extension_name: str) -> bool:
        """
        Check if the extension exists in memory (dict) or in the folder or in zarr.
        """
        if extension_name in self.extensions:
            return True
        elif self.format == "memory":
            return False
        elif extension_name in self.get_saved_extension_names():
            return True
        else:
            return False

    def get_computable_extensions(self):
        """
        Get all extensions that can be computed by the analyzer.
        """
        return get_available_analyzer_extensions()

    def get_default_extension_params(self, extension_name: str) -> dict:
        """
        Get the default params for an extension.

        Parameters
        ----------
        extension_name : str
            The extension name

        Returns
        -------
        default_params : dict
            The default parameters for the extension
        """
        return get_default_analyzer_extension_params(extension_name)


def _sort_extensions_by_dependency(extensions):
    """
    Sorts a dictionary of extensions so that the parents of each extension are on the "left" of their children.
    Assumes there is a valid ordering of the included extensions.

    Parameters
    ----------
    extensions : dict
        A dict of extensions.

    Returns
    -------
    sorted_extensions : dict
        A dict of extensions, with the parents on the left of their children.
    """

    extensions_list = list(extensions.keys())
    extension_params = list(extensions.values())

    i = 0
    while i < len(extensions_list):

        extension = extensions_list[i]
        dependencies = get_extension_class(extension).depend_on

        # Split cases with an "or" in them, and flatten into a list
        dependencies = list(chain.from_iterable([dependency.split("|") for dependency in dependencies]))

        # Should only iterate if nothing has happened.
        # Otherwise, should check the dependency which has just been moved => at position i
        did_nothing = True
        for dependency in dependencies:

            # if dependency is on the right, move it left of the current dependency
            if dependency in extensions_list[i:]:

                dependency_arg = extensions_list.index(dependency)

                extension_params.pop(dependency_arg)
                extension_params.insert(i, extensions[dependency])

                extensions_list.pop(dependency_arg)
                extensions_list.insert(i, dependency)

                did_nothing = False

        if did_nothing:
            i += 1

    return dict(zip(extensions_list, extension_params))


global _possible_extensions
_possible_extensions = []

global _extension_children
_extension_children = {}


def _get_children_dependencies(extension_name):
    """
    Extension classes have a `depend_on` attribute to declare on which class they
    depend on. For instance "templates" depends on "waveforms". "waveforms" depends on "random_spikes".

    This function is going the opposite way: it finds all children that depend on a
    particular extension.

    The implementation is recursive so that the output includes children, grand children, great grand children, etc.

    This function is useful for deleting existing extensions on recompute.
    For instance, recomputing the "waveforms" needs to delete the "templates", since the latter depends on the former.
    For this particular example, if we change the "ms_before" parameter of the "waveforms", also the "templates" will
    require recomputation as this parameter is inherited.
    """
    names = []
    children = _extension_children[extension_name]
    for child in children:
        if child not in names:
            names.append(child)
        grand_children = _get_children_dependencies(child)
        names.extend(grand_children)
    return list(names)


def register_result_extension(extension_class):
    """
    This maintains a list of possible extensions that are available.
    It depends on the imported submodules (e.g. for postprocessing module).

    For instance with:
    import spikeinterface as si
    only one extension will be available
    but with
    import spikeinterface.postprocessing
    more extensions will be available
    """
    assert issubclass(extension_class, AnalyzerExtension)
    assert extension_class.extension_name is not None, "extension_name must not be None"
    global _possible_extensions

    already_registered = any(extension_class is ext for ext in _possible_extensions)
    if not already_registered:
        assert all(
            extension_class.extension_name != ext.extension_name for ext in _possible_extensions
        ), "Extension name already exists"

        _possible_extensions.append(extension_class)

        # create the children dpendencies to be able to delete on re-compute
        _extension_children[extension_class.extension_name] = []
        for parent_name in extension_class.depend_on:
            if "|" in parent_name:
                for name in parent_name.split("|"):
                    _extension_children[name].append(extension_class.extension_name)
            else:
                _extension_children[parent_name].append(extension_class.extension_name)


def get_extension_class(extension_name: str, auto_import=True):
    """
    Get extension class from name and check if registered.

    Parameters
    ----------
    extension_name : str
        The extension name.
    auto_import : bool, default: True
        Auto import the module if the extension class is not registered yet.

    Returns
    -------
    ext_class:
        The class of the extension.
    """
    global _possible_extensions
    extensions_dict = {ext.extension_name: ext for ext in _possible_extensions}

    if extension_name not in extensions_dict:
        if extension_name in _builtin_extensions:
            module = _builtin_extensions[extension_name]
            if auto_import:
                imported_module = importlib.import_module(module)
                extensions_dict = {ext.extension_name: ext for ext in _possible_extensions}
            else:
                raise ValueError(
                    f"Extension '{extension_name}' is not registered, please import related module before use: 'import {module}'"
                )
        else:
            raise ValueError(f"Extension '{extension_name}' is unknown maybe this is an external extension or a typo.")

    ext_class = extensions_dict[extension_name]
    return ext_class


def get_available_analyzer_extensions():
    """
    Get all extensions that can be computed by the analyzer.
    """
    return list(_builtin_extensions.keys())


def get_default_analyzer_extension_params(extension_name: str):
    """
    Get the default params for an extension.

    Parameters
    ----------
    extension_name : str
        The extension name

    Returns
    -------
    default_params : dict
        The default parameters for the extension
    """
    import inspect

    extension_class = get_extension_class(extension_name)

    sig = inspect.signature(extension_class._set_params)
    default_params = {
        k: v.default for k, v in sig.parameters.items() if k != "self" and v.default != inspect.Parameter.empty
    }

    return default_params


class AnalyzerExtension:
    """
    This the base class to extend the SortingAnalyzer.
    It can handle persistency to disk for any computations related to:

    For instance:
      * waveforms
      * principal components
      * spike amplitudes
      * quality metrics

    Possible extension can be registered on-the-fly at import time with register_result_extension() mechanism.
    It also enables any custom computation on top of the SortingAnalyzer to be implemented by the user.

    An extension needs to inherit from this class and implement some attributes and abstract methods:

      * extension_name
      * depend_on
      * need_recording
      * use_nodepipeline
      * nodepipeline_variables only if use_nodepipeline=True
      * need_job_kwargs
      * _set_params()
      * _run()
      * _select_extension_data()
      * _merge_extension_data()
      * _get_data()

    The subclass must also set an `extension_name` class attribute which is not None by default.

    The subclass must also hanle an attribute `data` which is a dict contain the results after the `run()`.

    All AnalyzerExtension will have a function associate for instance (this use the function_factory):
    compute_unit_location(sorting_analyzer, ...) will be equivalent to sorting_analyzer.compute("unit_location", ...)


    """

    extension_name = None
    depend_on = []
    need_recording = False
    use_nodepipeline = False
    nodepipeline_variables = None
    need_job_kwargs = False
    need_backward_compatibility_on_load = False

    def __init__(self, sorting_analyzer):
        self._sorting_analyzer = weakref.ref(sorting_analyzer)

        self.params = None
        self.run_info = self._default_run_info_dict()
        self.data = dict()

    def _default_run_info_dict(self):
        return dict(run_completed=False, runtime_s=None)

    #######
    # This 3 methods must be implemented in the subclass!!!
    # See DummyAnalyzerExtension in test_sortinganalyzer.py as a simple example
    def _run(self, **kwargs):
        # must be implemented in subclass
        # must populate the self.data dictionary
        raise NotImplementedError

    def _set_params(self, **params):
        # must be implemented in subclass
        # must return a cleaned version of params dict
        raise NotImplementedError

    def _select_extension_data(self, unit_ids):
        # must be implemented in subclass
        raise NotImplementedError

    def _merge_extension_data(
        self, merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask, verbose=False, **job_kwargs
    ):
        # must be implemented in subclass
        raise NotImplementedError

    def _split_extension_data(self, split_units, new_unit_ids, new_sorting_analyzer, verbose=False, **job_kwargs):
        # must be implemented in subclass
        raise NotImplementedError

    def _get_pipeline_nodes(self):
        # must be implemented in subclass only if use_nodepipeline=True
        raise NotImplementedError

    def _get_data(self):
        # must be implemented in subclass
        raise NotImplementedError

    def _handle_backward_compatibility_on_load(self):
        # must be implemented in subclass only if need_backward_compatibility_on_load=True
        raise NotImplementedError

    @classmethod
    def function_factory(cls):
        # make equivalent
        # comptute_unit_location(sorting_analyzer, ...) <> sorting_analyzer.compute("unit_location", ...)
        # this also make backcompatibility
        # comptute_unit_location(we, ...)

        class FuncWrapper:
            def __init__(self, extension_name):
                self.extension_name = extension_name

            def __call__(self, sorting_analyzer, load_if_exists=None, *args, **kwargs):
                from .waveforms_extractor_backwards_compatibility import MockWaveformExtractor

                if isinstance(sorting_analyzer, MockWaveformExtractor):
                    # backward compatibility with WaveformsExtractor
                    sorting_analyzer = sorting_analyzer.sorting_analyzer

                if not isinstance(sorting_analyzer, SortingAnalyzer):
                    raise ValueError(f"compute_{self.extension_name}() needs a SortingAnalyzer instance")

                if load_if_exists is not None:
                    # backward compatibility with "load_if_exists"
                    warnings.warn(
                        f"compute_{cls.extension_name}(..., load_if_exists=True/False) is kept for backward compatibility but should not be used anymore"
                    )
                    assert isinstance(load_if_exists, bool)
                    if load_if_exists:
                        ext = sorting_analyzer.get_extension(self.extension_name)
                        return ext

                ext = sorting_analyzer.compute(cls.extension_name, *args, **kwargs)
                return ext.get_data()

        func = FuncWrapper(cls.extension_name)
        func.__doc__ = cls.__doc__
        return func

    @property
    def sorting_analyzer(self):
        # Important : to avoid the SortingAnalyzer referencing a AnalyzerExtension
        # and AnalyzerExtension referencing a SortingAnalyzer we need a weakref.
        # Otherwise the garbage collector is not working properly.
        # and so the SortingAnalyzer + its recording are still alive even after deleting explicitly
        # the SortingAnalyzer which makes it impossible to delete the folder when using memmap.
        sorting_analyzer = self._sorting_analyzer()
        if sorting_analyzer is None:
            raise ValueError(f"The extension {self.extension_name} has lost its SortingAnalyzer")
        return sorting_analyzer

    # some attribuites come from sorting_analyzer
    @property
    def format(self):
        return self.sorting_analyzer.format

    @property
    def sparsity(self):
        return self.sorting_analyzer.sparsity

    @property
    def folder(self):
        return self.sorting_analyzer.folder

    def _get_binary_extension_folder(self):
        extension_folder = self.folder / "extensions" / self.extension_name
        return extension_folder

    def _get_zarr_extension_group(self, mode="r+"):
        zarr_root = self.sorting_analyzer._get_zarr_root(mode=mode)
        extension_group = zarr_root["extensions"][self.extension_name]
        return extension_group

    @classmethod
    def load(cls, sorting_analyzer):
        ext = cls(sorting_analyzer)
        ext.load_params()
        ext.load_run_info()
        if ext.run_info is not None:
            if ext.run_info["run_completed"]:
                ext.load_data()
                if cls.need_backward_compatibility_on_load:
                    ext._handle_backward_compatibility_on_load()
                if len(ext.data) > 0:
                    return ext
        else:
            # this is for back-compatibility of old analyzers
            ext.load_data()
            if cls.need_backward_compatibility_on_load:
                ext._handle_backward_compatibility_on_load()
            if len(ext.data) > 0:
                return ext
        # If extension run not completed, or data has gone missing,
        # return None to indicate that the extension should be (re)computed.
        return None

    def load_run_info(self):
        run_info = None
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            run_info_file = extension_folder / "run_info.json"
            if run_info_file.is_file():
                with open(str(run_info_file), "r") as f:
                    run_info = json.load(f)

        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r")
            run_info = extension_group.attrs.get("run_info", None)

        if run_info is None:
            warnings.warn(f"Found no run_info file for {self.extension_name}, extension should be re-computed.")
        self.run_info = run_info

    def load_params(self):
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            params_file = extension_folder / "params.json"
            assert params_file.is_file(), f"No params file in extension {self.extension_name} folder"
            with open(str(params_file), "r") as f:
                params = json.load(f)

        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r")
            assert "params" in extension_group.attrs, f"No params file in extension {self.extension_name} folder"
            params = extension_group.attrs["params"]

        self.params = params

    def load_data(self):
        ext_data = None
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            for ext_data_file in extension_folder.iterdir():
                # patch for https://github.com/SpikeInterface/spikeinterface/issues/3041
                # maybe add a check for version number from the info.json during loading only
                if (
                    ext_data_file.name == "params.json"
                    or ext_data_file.name == "info.json"
                    or ext_data_file.name == "run_info.json"
                    or str(ext_data_file.name).startswith("._")  # ignore AppleDouble format files
                ):
                    continue
                ext_data_name = ext_data_file.stem
                if ext_data_file.suffix == ".json":
                    with ext_data_file.open("r") as f:
                        ext_data = json.load(f)
                elif ext_data_file.suffix == ".npy":
                    # The lazy loading of an extension is complicated because if we compute again
                    # and have a link to the old buffer on windows then it fails
                    # ext_data = np.load(ext_data_file, mmap_mode="r")
                    # so we go back to full loading
                    ext_data = np.load(ext_data_file)
                elif ext_data_file.suffix == ".csv":
                    import pandas as pd

                    ext_data = pd.read_csv(ext_data_file, index_col=0)
                    # we need to cast the index to the unit id dtype (int or str)
                    unit_ids = self.sorting_analyzer.unit_ids
                    if ext_data.shape[0] == unit_ids.size:
                        # we force dtype to be the same as unit_ids
                        if ext_data.index.dtype != unit_ids.dtype:
                            ext_data.index = ext_data.index.astype(unit_ids.dtype)

                elif ext_data_file.suffix == ".pkl":
                    with ext_data_file.open("rb") as f:
                        ext_data = pickle.load(f)
                else:
                    continue
                self.data[ext_data_name] = ext_data

        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r")
            for ext_data_name in extension_group.keys():
                ext_data_ = extension_group[ext_data_name]
                if "dict" in ext_data_.attrs:
                    ext_data = ext_data_[0]
                elif "dataframe" in ext_data_.attrs:
                    import pandas as pd

                    index = ext_data_["index"]
                    ext_data = pd.DataFrame(index=index)
                    for col in ext_data_.keys():
                        if col != "index":
                            ext_data.loc[:, col] = ext_data_[col][:]
                    ext_data = ext_data.convert_dtypes()
                elif "object" in ext_data_.attrs:
                    ext_data = ext_data_[0]
                else:
                    # this load in memmory
                    ext_data = np.array(ext_data_)
                self.data[ext_data_name] = ext_data

        if len(self.data) == 0:
            warnings.warn(f"Found no data for {self.extension_name}, extension should be re-computed.")

    def copy(self, new_sorting_analyzer, unit_ids=None):
        # alessio : please note that this also replace the old select_units!!!
        new_extension = self.__class__(new_sorting_analyzer)
        new_extension.params = self.params.copy()
        if unit_ids is None:
            new_extension.data = self.data
        else:
            new_extension.data = self._select_extension_data(unit_ids)
        new_extension.run_info = copy(self.run_info)
        new_extension.save()
        return new_extension

    def merge(
        self,
        new_sorting_analyzer,
        merge_unit_groups,
        new_unit_ids,
        keep_mask=None,
        verbose=False,
        **job_kwargs,
    ):
        new_extension = self.__class__(new_sorting_analyzer)
        new_extension.params = self.params.copy()
        new_extension.data = self._merge_extension_data(
            merge_unit_groups, new_unit_ids, new_sorting_analyzer, keep_mask, verbose=verbose, **job_kwargs
        )
        new_extension.run_info = copy(self.run_info)
        new_extension.save()
        return new_extension

    def split(
        self,
        new_sorting_analyzer,
        split_units,
        new_unit_ids,
        verbose=False,
        **job_kwargs,
    ):
        new_extension = self.__class__(new_sorting_analyzer)
        new_extension.params = self.params.copy()
        new_extension.data = self._split_extension_data(
            split_units, new_unit_ids, new_sorting_analyzer, verbose=verbose, **job_kwargs
        )
        new_extension.run_info = copy(self.run_info)
        new_extension.save()
        return new_extension

    def run(self, save=True, **kwargs):
        if save and not self.sorting_analyzer.is_read_only():
            # NB: this call to _save_params() also resets the folder or zarr group
            self._save_params()
            self._save_importing_provenance()

        t_start = perf_counter()
        self._run(**kwargs)
        t_end = perf_counter()
        self.run_info["runtime_s"] = t_end - t_start
        self.run_info["run_completed"] = True

        if save and not self.sorting_analyzer.is_read_only():
            self._save_run_info()
            self._save_data()
            if self.format == "zarr":
                import zarr

                zarr.consolidate_metadata(self.sorting_analyzer._get_zarr_root().store)

    def save(self):
        self._save_params()
        self._save_importing_provenance()
        self._save_run_info()
        self._save_data()

        if self.format == "zarr":
            import zarr

            zarr.consolidate_metadata(self.sorting_analyzer._get_zarr_root().store)

    def _save_data(self):
        if self.format == "memory":
            return

        if self.sorting_analyzer.is_read_only():
            raise ValueError(f"The SortingAnalyzer is read-only saving extension {self.extension_name} is not possible")

        try:
            # pandas is a weak dependency for spikeinterface.core
            import pandas as pd

            HAS_PANDAS = True
        except:
            HAS_PANDAS = False

        if self.format == "binary_folder":

            extension_folder = self._get_binary_extension_folder()
            for ext_data_name, ext_data in self.data.items():
                if isinstance(ext_data, dict):
                    with (extension_folder / f"{ext_data_name}.json").open("w") as f:
                        json.dump(ext_data, f)
                elif isinstance(ext_data, np.ndarray):
                    data_file = extension_folder / f"{ext_data_name}.npy"
                    if isinstance(ext_data, np.memmap) and data_file.exists():
                        # important some SortingAnalyzer like ComputeWaveforms already run the computation with memmap
                        # so no need to save theses array
                        pass
                    else:
                        np.save(data_file, ext_data)
                elif HAS_PANDAS and isinstance(ext_data, pd.DataFrame):
                    ext_data.to_csv(extension_folder / f"{ext_data_name}.csv", index=True)
                else:
                    try:
                        with (extension_folder / f"{ext_data_name}.pkl").open("wb") as f:
                            pickle.dump(ext_data, f)
                    except:
                        raise Exception(f"Could not save {ext_data_name} as extension data")
        elif self.format == "zarr":
            import numcodecs

            saving_options = self.sorting_analyzer._backend_options.get("saving_options", {})
            extension_group = self._get_zarr_extension_group(mode="r+")

            # if compression is not externally given, we use the default
            if "compressor" not in saving_options:
                saving_options["compressor"] = get_default_zarr_compressor()

            for ext_data_name, ext_data in self.data.items():
                if ext_data_name in extension_group:
                    del extension_group[ext_data_name]
                if isinstance(ext_data, dict):
                    extension_group.create_dataset(
                        name=ext_data_name, data=np.array([ext_data], dtype=object), object_codec=numcodecs.JSON()
                    )
                elif isinstance(ext_data, np.ndarray):
                    extension_group.create_dataset(name=ext_data_name, data=ext_data, **saving_options)
                elif HAS_PANDAS and isinstance(ext_data, pd.DataFrame):
                    df_group = extension_group.create_group(ext_data_name)
                    # first we save the index
                    indices = ext_data.index.to_numpy()
                    if indices.dtype.kind == "O":
                        indices = indices.astype(str)
                    df_group.create_dataset(name="index", data=indices)
                    for col in ext_data.columns:
                        col_data = ext_data[col].to_numpy()
                        if col_data.dtype.kind == "O":
                            col_data = col_data.astype(str)
                        df_group.create_dataset(name=col, data=col_data)
                    df_group.attrs["dataframe"] = True
                else:
                    # any object
                    try:
                        extension_group.create_dataset(
                            name=ext_data_name, data=np.array([ext_data], dtype=object), object_codec=numcodecs.Pickle()
                        )
                    except:
                        raise Exception(f"Could not save {ext_data_name} as extension data")
                    extension_group[ext_data_name].attrs["object"] = True

    def _reset_extension_folder(self):
        """
        Delete the extension in a folder (binary or zarr) and create an empty one.
        """
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            if extension_folder.is_dir():
                shutil.rmtree(extension_folder)
            extension_folder.mkdir(exist_ok=False, parents=True)

        elif self.format == "zarr":
            import zarr

            zarr_root = self.sorting_analyzer._get_zarr_root(mode="r+")
            _ = zarr_root["extensions"].create_group(self.extension_name, overwrite=True)
            zarr.consolidate_metadata(zarr_root.store)

    def _delete_extension_folder(self):
        """
        Delete the extension in a folder (binary or zarr).
        """
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            if extension_folder.is_dir():
                shutil.rmtree(extension_folder)

        elif self.format == "zarr":
            import zarr

            zarr_root = self.sorting_analyzer._get_zarr_root(mode="r+")
            if self.extension_name in zarr_root["extensions"]:
                del zarr_root["extensions"][self.extension_name]
                zarr.consolidate_metadata(zarr_root.store)

    def delete(self):
        """
        Delete the extension from the folder or zarr and from the dict.
        """
        self._delete_extension_folder()
        self.params = None
        self.run_info = self._default_run_info_dict()
        self.data = dict()

    def reset(self):
        """
        Reset the extension.
        Delete the sub folder and create a new empty one.
        """
        self._reset_extension_folder()
        self.params = None
        self.run_info = self._default_run_info_dict()
        self.data = dict()

    def set_params(self, save=True, **params):
        """
        Set parameters for the extension and
        make it persistent in json.
        """
        # this ensure data is also deleted and corresponds to params
        # this also ensure the group is created
        if save:
            self._reset_extension_folder()

        params = self._set_params(**params)
        self.params = params

        if self.sorting_analyzer.is_read_only():
            return

        if save:
            self._save_params()
            self._save_importing_provenance()

    def _save_params(self):
        params_to_save = self.params.copy()

        self._reset_extension_folder()

        # TODO make sparsity local Result specific
        # if "sparsity" in params_to_save and params_to_save["sparsity"] is not None:
        #     assert isinstance(
        #         params_to_save["sparsity"], ChannelSparsity
        #     ), "'sparsity' parameter must be a ChannelSparsity object!"
        #     params_to_save["sparsity"] = params_to_save["sparsity"].to_dict()

        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            extension_folder.mkdir(exist_ok=True, parents=True)
            param_file = extension_folder / "params.json"
            param_file.write_text(json.dumps(check_json(params_to_save), indent=4), encoding="utf8")
        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r+")
            extension_group.attrs["params"] = check_json(params_to_save)

    def _save_importing_provenance(self):
        # this saves the class info, this is not uselful at the moment but could be useful in future
        # if some class changes the data model and if we need to make backwards compatibility
        # we have the same machanism in base.py for recording and sorting

        info = retrieve_importing_provenance(self.__class__)
        if self.format == "binary_folder":
            extension_folder = self._get_binary_extension_folder()
            extension_folder.mkdir(exist_ok=True, parents=True)
            info_file = extension_folder / "info.json"
            info_file.write_text(json.dumps(info, indent=4), encoding="utf8")
        elif self.format == "zarr":
            extension_group = self._get_zarr_extension_group(mode="r+")
            extension_group.attrs["info"] = info

    def _save_run_info(self):
        if self.run_info is not None:
            run_info = self.run_info.copy()

            if self.format == "binary_folder":
                extension_folder = self._get_binary_extension_folder()
                run_info_file = extension_folder / "run_info.json"
                run_info_file.write_text(json.dumps(run_info, indent=4), encoding="utf8")
            elif self.format == "zarr":
                extension_group = self._get_zarr_extension_group(mode="r+")
                extension_group.attrs["run_info"] = run_info

    def get_pipeline_nodes(self):
        assert (
            self.use_nodepipeline
        ), "AnalyzerExtension.get_pipeline_nodes() must be called only when use_nodepipeline=True"
        return self._get_pipeline_nodes()

    def get_data(self, *args, **kwargs):
        if self.run_info is not None:
            assert self.run_info[
                "run_completed"
            ], f"You must run the extension {self.extension_name} before retrieving data"
        assert len(self.data) > 0, "Extension has been run but no data found."
        return self._get_data(*args, **kwargs)


# this is a hardcoded list to to improve error message and auto_import mechanism
# this is important because extension are registered when the submodule is imported
_builtin_extensions = {
    # from core
    "random_spikes": "spikeinterface.core",
    "waveforms": "spikeinterface.core",
    "templates": "spikeinterface.core",
    # "fast_templates": "spikeinterface.core",
    "noise_levels": "spikeinterface.core",
    # from postprocessing
    "amplitude_scalings": "spikeinterface.postprocessing",
    "correlograms": "spikeinterface.postprocessing",
    "isi_histograms": "spikeinterface.postprocessing",
    "principal_components": "spikeinterface.postprocessing",
    "spike_amplitudes": "spikeinterface.postprocessing",
    "spike_locations": "spikeinterface.postprocessing",
    "template_metrics": "spikeinterface.postprocessing",
    "template_similarity": "spikeinterface.postprocessing",
    "unit_locations": "spikeinterface.postprocessing",
    # from quality metrics
    "quality_metrics": "spikeinterface.qualitymetrics",
}
