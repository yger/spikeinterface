"""
test for BaseRecording are done with BinaryRecordingExtractor.
but check only for BaseRecording general methods.
"""

import json
import pickle
from pathlib import Path
import pytest
import numpy as np
from numpy.testing import assert_raises

from probeinterface import Probe, ProbeGroup, generate_linear_probe

from spikeinterface.core import BinaryRecordingExtractor, NumpyRecording, load, get_default_zarr_compressor
from spikeinterface.core.base import BaseExtractor
from spikeinterface.core.testing import check_recordings_equal

from spikeinterface.core import generate_recording


def test_BaseRecording(create_cache_folder):
    cache_folder = create_cache_folder
    num_seg = 2
    num_chan = 3
    num_samples = 30
    sampling_frequency = 10000
    dtype = "int16"

    file_paths = [cache_folder / f"test_base_recording_{i}.raw" for i in range(num_seg)]
    for i in range(num_seg):
        a = np.memmap(file_paths[i], dtype=dtype, mode="w+", shape=(num_samples, num_chan))
        a[:] = np.random.randn(*a.shape).astype(dtype)
    rec = BinaryRecordingExtractor(
        file_paths=file_paths, sampling_frequency=sampling_frequency, num_channels=num_chan, dtype=dtype
    )

    assert rec.get_num_segments() == 2
    assert rec.get_num_channels() == 3

    assert np.all(rec.ids_to_indices([0, 1, 2]) == [0, 1, 2])
    assert np.all(rec.ids_to_indices([0, 1, 2], prefer_slice=True) == slice(0, 3, None))

    # annotations / properties
    rec.annotate(yep="yop")
    assert rec.get_annotation("yep") == "yop"

    rec.set_channel_groups([0, 0, 1])

    rec.set_property("quality", [1.0, 3.3, np.nan])
    values = rec.get_property("quality")
    assert np.all(
        values[:2]
        == [
            1.0,
            3.3,
        ]
    )

    # missing property
    rec.set_property("string_property", ["ciao", "bello"], ids=[0, 1])
    values = rec.get_property("string_property")
    assert values[2] == ""

    # setting an different type raises an error
    assert_raises(
        Exception,
        rec.set_property,
        key="string_property_nan",
        values=["ciao", "bello"],
        ids=[0, 1],
        missing_value=np.nan,
    )

    # int properties without missing values raise an error
    assert_raises(Exception, rec.set_property, key="int_property", values=[5, 6], ids=[1, 2])

    rec.set_property("int_property", [5, 6], ids=[1, 2], missing_value=200)
    values = rec.get_property("int_property")
    assert values.dtype.kind == "i"

    times0 = rec.get_times(segment_index=0)

    # dump/load dict
    d = rec.to_dict(include_annotations=True, include_properties=True)
    rec2 = BaseExtractor.from_dict(d)
    rec3 = load(d)
    check_recordings_equal(rec, rec2, return_in_uV=False, check_annotations=True, check_properties=True)
    check_recordings_equal(rec, rec3, return_in_uV=False, check_annotations=True, check_properties=True)

    # dump/load json
    rec.dump_to_json(cache_folder / "test_BaseRecording.json")
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording.json")
    rec3 = load(cache_folder / "test_BaseRecording.json")
    check_recordings_equal(rec, rec2, return_in_uV=False, check_annotations=True, check_properties=False)
    check_recordings_equal(rec, rec3, return_in_uV=False, check_annotations=True, check_properties=False)

    # dump/load pickle
    rec.dump_to_pickle(cache_folder / "test_BaseRecording.pkl")
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording.pkl")
    rec3 = load(cache_folder / "test_BaseRecording.pkl")
    check_recordings_equal(rec, rec2, return_in_uV=False, check_annotations=True, check_properties=True)
    check_recordings_equal(rec, rec3, return_in_uV=False, check_annotations=True, check_properties=True)

    # dump/load dict - relative
    d = rec.to_dict(relative_to=cache_folder, recursive=True)
    rec2 = BaseExtractor.from_dict(d, base_folder=cache_folder)
    rec3 = load(d, base_folder=cache_folder)

    # dump/load json - relative to
    rec.dump_to_json(cache_folder / "test_BaseRecording_rel.json", relative_to=cache_folder)
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording_rel.json", base_folder=cache_folder)
    rec3 = load(cache_folder / "test_BaseRecording_rel.json", base_folder=cache_folder)

    # dump/load relative=True
    rec.dump_to_json(cache_folder / "test_BaseRecording_rel_true.json", relative_to=True)
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording_rel_true.json", base_folder=True)
    rec3 = load(cache_folder / "test_BaseRecording_rel_true.json", base_folder=True)
    check_recordings_equal(rec, rec2, return_in_uV=False, check_annotations=True)
    check_recordings_equal(rec, rec3, return_in_uV=False, check_annotations=True)
    with open(cache_folder / "test_BaseRecording_rel_true.json") as json_file:
        data = json.load(json_file)
        assert (
            "/" not in data["kwargs"]["file_paths"][0]
        )  # Relative to parent folder, so there shouldn't be any '/' in the path.

    # dump/load pkl - relative to
    rec.dump_to_pickle(cache_folder / "test_BaseRecording_rel.pkl", relative_to=cache_folder)
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording_rel.pkl", base_folder=cache_folder)
    rec3 = load(cache_folder / "test_BaseRecording_rel.pkl", base_folder=cache_folder)

    # dump/load relative=True
    rec.dump_to_pickle(cache_folder / "test_BaseRecording_rel_true.pkl", relative_to=True)
    rec2 = BaseExtractor.load(cache_folder / "test_BaseRecording_rel_true.pkl", base_folder=True)
    rec3 = load(cache_folder / "test_BaseRecording_rel_true.pkl", base_folder=True)
    check_recordings_equal(rec, rec2, return_in_uV=False, check_annotations=True)
    check_recordings_equal(rec, rec3, return_in_uV=False, check_annotations=True)
    with open(cache_folder / "test_BaseRecording_rel_true.pkl", "rb") as pkl_file:
        data = pickle.load(pkl_file)
        assert (
            "/" not in data["kwargs"]["file_paths"][0]
        )  # Relative to parent folder, so there shouldn't be any '/' in the path.

    # cache to binary
    folder = cache_folder / "simple_recording"
    rec.save(format="binary", folder=folder)
    rec2 = BaseExtractor.load_from_folder(folder)
    assert "quality" in rec2.get_property_keys()
    values = rec2.get_property("quality")
    assert values[0] == 1.0
    assert values[1] == 3.3
    assert np.isnan(values[2])

    groups = rec2.get_channel_groups()
    assert np.array_equal(groups, [0, 0, 1])

    # but also possible
    rec3 = BaseExtractor.load(cache_folder / "simple_recording")

    # cache to memory
    rec4 = rec3.save(format="memory", shared=False)
    traces4 = rec4.get_traces(segment_index=0)
    traces = rec.get_traces(segment_index=0)
    assert np.array_equal(traces4, traces)

    # cache to sharedmemory
    rec5 = rec3.save(format="memory", shared=True)
    traces5 = rec5.get_traces(segment_index=0)
    traces = rec.get_traces(segment_index=0)
    assert np.array_equal(traces5, traces)

    # cache joblib several jobs
    folder = cache_folder / "simple_recording2"
    rec2 = rec.save(format="binary", folder=folder, chunk_size=10, n_jobs=4)
    traces2 = rec2.get_traces(segment_index=0)

    # set/get Probe only 2 channels
    probe = Probe(ndim=2)
    positions = [[0.0, 0.0], [0.0, 15.0], [0, 30.0]]
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices([2, -1, 0])
    probe.create_auto_shape()

    rec_p = rec.set_probe(probe, group_mode="by_shank")
    rec_p = rec.set_probe(probe, group_mode="by_probe")
    positions2 = rec_p.get_channel_locations()
    assert np.array_equal(positions2, [[0, 30.0], [0.0, 0.0]])

    probe2 = rec_p.get_probe()
    positions3 = probe2.contact_positions
    assert np.array_equal(positions2, positions3)

    assert np.array_equal(probe2.device_channel_indices, [0, 1])

    # test save with probe
    folder = cache_folder / "simple_recording3"
    rec2 = rec_p.save(folder=folder, chunk_size=10, n_jobs=2)
    rec2 = load(folder)
    probe2 = rec2.get_probe()
    assert np.array_equal(probe2.contact_positions, [[0, 30.0], [0.0, 0.0]])
    positions2 = rec_p.get_channel_locations()
    assert np.array_equal(positions2, [[0, 30.0], [0.0, 0.0]])
    traces2 = rec2.get_traces(segment_index=0)
    assert np.array_equal(traces2, rec_p.get_traces(segment_index=0))

    # from probeinterface.plotting import plot_probe_group, plot_probe
    # import matplotlib.pyplot as plt
    # plot_probe(probe)
    # plot_probe(probe2)
    # plt.show()

    # set unconnected probe
    probe = Probe(ndim=2)
    positions = [[0.0, 0.0], [0.0, 15.0], [0, 30.0]]
    probe.set_contacts(positions=positions, shapes="circle", shape_params={"radius": 5})
    probe.set_device_channel_indices([-1, -1, -1])
    probe.create_auto_shape()

    rec_empty_probe = rec.set_probe(probe, group_mode="by_shank")
    assert rec_empty_probe.channel_ids.size == 0

    # test scaling parameters
    sampling_frequency = 30000
    traces = np.zeros((1000, 5), dtype="int16")
    rec_int16 = NumpyRecording([traces], sampling_frequency)
    assert rec_int16.get_dtype() == "int16"

    traces = np.zeros((1000, 5), dtype="uint16")
    rec_uint16 = NumpyRecording([traces], sampling_frequency)
    assert rec_uint16.get_dtype() == "uint16"

    traces_int16 = rec_int16.get_traces()
    assert traces_int16.dtype == "int16"

    # Both return_scaled and return_in_uV raise error when no gain_to_uV/offset_to_uV properties
    with pytest.raises(ValueError):
        traces_float32 = rec_int16.get_traces(return_scaled=True)
    with pytest.raises(ValueError):
        traces_float32 = rec_int16.get_traces(return_in_uV=True)

    # Set properties and test both parameters
    rec_int16.set_property("gain_to_uV", [0.195] * 5)
    rec_int16.set_property("offset_to_uV", [0.0] * 5)

    # Test deprecated return_scaled parameter
    traces_float32_old = rec_int16.get_traces(return_scaled=True)  # Keep this for testing the deprecation warning
    assert traces_float32_old.dtype == "float32"

    # Test new return_in_uV parameter
    traces_float32_new = rec_int16.get_traces(return_in_uV=True)
    assert traces_float32_new.dtype == "float32"

    # Verify both parameters produce the same result
    assert np.array_equal(traces_float32_old, traces_float32_new)

    # test cast with dtype
    rec_float32 = rec_int16.astype("float32")
    assert rec_float32.get_dtype() == "float32"
    assert np.dtype(rec_float32.get_traces().dtype) == np.float32

    # test with t_start
    rec = BinaryRecordingExtractor(
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        num_channels=num_chan,
        dtype=dtype,
        t_starts=np.arange(num_seg) * 10.0,
    )
    times1 = rec.get_times(1)
    folder = cache_folder / "recording_with_t_start"
    rec2 = rec.save(folder=folder)
    assert np.allclose(times1, rec2.get_times(1))

    # test with time_vector
    rec = BinaryRecordingExtractor(
        file_paths=file_paths,
        sampling_frequency=sampling_frequency,
        num_channels=num_chan,
        dtype=dtype,
    )
    rec.set_times(np.arange(num_samples) / sampling_frequency + 30.0, segment_index=0)
    rec.set_times(np.arange(num_samples) / sampling_frequency + 40.0, segment_index=1)
    times1 = rec.get_times(1)
    folder = cache_folder / "recording_with_times"
    rec2 = rec.save(folder=folder)
    assert np.allclose(times1, rec2.get_times(1))
    rec3 = load(folder)
    assert np.allclose(times1, rec3.get_times(1))

    # reset times
    rec.reset_times()
    for segm in range(num_seg):
        time_info = rec.get_time_info(segment_index=segm)
        assert not rec.has_time_vector(segment_index=segm)
        assert time_info["t_start"] is None
        assert time_info["time_vector"] is None
        assert time_info["sampling_frequency"] == rec.sampling_frequency

    # resetting time again should be ok
    rec.reset_times()

    # test 3d probe
    rec_3d = generate_recording(ndim=3, num_channels=30)
    locations_3d = rec_3d.get_property("location")

    locations_xy = rec_3d.get_channel_locations(axes="xy")
    assert np.allclose(locations_xy, locations_3d[:, [0, 1]])

    locations_xz = rec_3d.get_channel_locations(axes="xz")
    assert np.allclose(locations_xz, locations_3d[:, [0, 2]])

    locations_zy = rec_3d.get_channel_locations(axes="zy")
    assert np.allclose(locations_zy, locations_3d[:, [2, 1]])

    locations_xzy = rec_3d.get_channel_locations(axes="xzy")
    assert np.allclose(locations_xzy, locations_3d[:, [0, 2, 1]])

    rec_2d = rec_3d.planarize(axes="zy")
    assert np.allclose(rec_2d.get_channel_locations(), locations_3d[:, [2, 1]])

    # test save to zarr
    compressor = get_default_zarr_compressor()
    rec_zarr = rec2.save(format="zarr", folder=cache_folder / "recording", compressor=compressor)
    rec_zarr_loaded = load(cache_folder / "recording.zarr")
    # annotations is False because Zarr adds compression ratios
    check_recordings_equal(rec2, rec_zarr, return_in_uV=False, check_annotations=False, check_properties=True)
    check_recordings_equal(
        rec_zarr, rec_zarr_loaded, return_in_uV=False, check_annotations=False, check_properties=True
    )
    for annotation_name in rec2.get_annotation_keys():
        assert rec2.get_annotation(annotation_name) == rec_zarr.get_annotation(annotation_name)
        assert rec2.get_annotation(annotation_name) == rec_zarr_loaded.get_annotation(annotation_name)

    rec_zarr2 = rec2.save(
        format="zarr", folder=cache_folder / "recording_channel_chunk", compressor=compressor, channel_chunk_size=2
    )
    rec_zarr2_loaded = load(cache_folder / "recording_channel_chunk.zarr")

    # annotations is False because Zarr adds compression ratios
    check_recordings_equal(rec2, rec_zarr2, return_in_uV=False, check_annotations=False, check_properties=True)
    check_recordings_equal(
        rec_zarr2, rec_zarr2_loaded, return_in_uV=False, check_annotations=False, check_properties=True
    )
    for annotation_name in rec2.get_annotation_keys():
        assert rec2.get_annotation(annotation_name) == rec_zarr2.get_annotation(annotation_name)
        assert rec2.get_annotation(annotation_name) == rec_zarr2_loaded.get_annotation(annotation_name)


def test_json_pickle_equivalence(create_cache_folder):
    """
    For a json-ifyable recording, the json and pickle outputs created by `dump` should be the same
    (except for the probe information). We check this here for a saved-then-loaded recording,
    which tests if relative paths are dealt with in the same way.
    """

    rec = generate_recording(durations=[1])
    cache_folder = create_cache_folder

    json_file_path = cache_folder / "recording.json"
    pkl_file_path = cache_folder / "recording.pkl"

    rec.dump(json_file_path, relative_to=cache_folder)
    rec.dump(pkl_file_path, relative_to=cache_folder)

    with open(json_file_path, "r") as f:
        data_json = json.load(f)

    with open(pkl_file_path, "rb") as f:
        data_pickle = pickle.load(f)

    for key, value in data_json.items():
        # skip probe info, since pickle keeps some additional information
        if key not in ["properties"]:
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    assert np.all(sub_value == data_pickle[key][sub_key])
            else:
                assert np.all(value == data_pickle[key])


def test_interleaved_probegroups():
    recording = generate_recording(durations=[1.0], num_channels=16)

    probe1 = generate_linear_probe(num_elec=8, ypitch=20.0)
    probe2_overlap = probe1.copy()

    probegroup_overlap = ProbeGroup()
    probegroup_overlap.add_probe(probe1)
    probegroup_overlap.add_probe(probe2_overlap)
    probegroup_overlap.set_global_device_channel_indices(np.arange(16))

    # setting overlapping probes should raise an error
    with pytest.raises(Exception):
        recording.set_probegroup(probegroup_overlap)

    probe2 = probe1.copy()
    probe2.move([100.0, 100.0])
    probegroup = ProbeGroup()
    probegroup.add_probe(probe1)
    probegroup.add_probe(probe2)
    probegroup.set_global_device_channel_indices(np.random.permutation(16))

    recording.set_probegroup(probegroup)
    probegroup_set = recording.get_probegroup()
    # check that the probe group is correctly set, by sorting the device channel indices
    assert np.array_equal(probegroup_set.get_global_device_channel_indices()["device_channel_indices"], np.arange(16))


def test_rename_channels():
    recording = generate_recording(durations=[1.0], num_channels=3)
    renamed_recording = recording.rename_channels(new_channel_ids=["a", "b", "c"])
    renamed_channel_ids = renamed_recording.get_channel_ids()
    assert np.array_equal(renamed_channel_ids, ["a", "b", "c"])


def test_select_channels():
    recording = generate_recording(durations=[1.0], num_channels=3)
    renamed_recording = recording.rename_channels(new_channel_ids=["a", "b", "c"])
    selected_recording = renamed_recording.select_channels(channel_ids=["a", "c"])
    selected_channel_ids = selected_recording.get_channel_ids()
    assert np.array_equal(selected_channel_ids, ["a", "c"])


def test_time_slice():
    # Case with sampling frequency
    sampling_frequency = 10_000.0
    recording = generate_recording(durations=[1.0], num_channels=3, sampling_frequency=sampling_frequency)

    sliced_recording_times = recording.time_slice(start_time=0.1, end_time=0.8)
    sliced_recording_frames = recording.frame_slice(start_frame=1000, end_frame=8000)

    assert np.allclose(sliced_recording_times.get_traces(), sliced_recording_frames.get_traces())


def test_out_of_range_time_slice():
    recording = generate_recording(durations=[0.100])  # duration = 0.1 s
    recording.shift_times(1.0)  # shifts start time to 1.0 s, end time to 1.1 s

    # start_time before recording
    with pytest.raises(ValueError, match="start_time .* is before the start time"):
        recording.time_slice(start_time=0, end_time=None)

    # end_time before start of recording
    with pytest.raises(ValueError, match="end_time .* is before the start time"):
        recording.time_slice(start_time=None, end_time=0.5)

    # start_time after end of recording
    with pytest.raises(ValueError, match="start_time .* is after the end time"):
        recording.time_slice(start_time=2.0, end_time=None)

    # end_time after end of recording
    with pytest.raises(ValueError, match="end_time .* is after the end time"):
        recording.time_slice(start_time=None, end_time=2.0)


def test_time_slice_with_time_vector():

    # Case with time vector
    sampling_frequency = 10_000.0
    recording = generate_recording(durations=[1.0], num_channels=3, sampling_frequency=sampling_frequency)
    times = 1 + np.arange(0, 10_000) / sampling_frequency
    recording.set_times(times=times, segment_index=0, with_warning=False)

    sliced_recording_times = recording.time_slice(start_time=1.1, end_time=1.8)
    sliced_recording_frames = recording.frame_slice(start_frame=1000, end_frame=8000)

    assert np.allclose(sliced_recording_times.get_traces(), sliced_recording_frames.get_traces())


if __name__ == "__main__":
    # test_BaseRecording()
    test_interleaved_probegroups()
