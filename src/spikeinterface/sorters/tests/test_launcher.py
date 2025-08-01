import sys
import shutil
import tempfile
import time
import pytest
from pathlib import Path
from platform import system
from spikeinterface import generate_ground_truth_recording
from spikeinterface.sorters import run_sorter_jobs, run_sorter_by_property


# no need to have many
NUM_RECORDINGS = 2
SORTERS = ["tridesclous2"]


def create_recordings(NUM_RECORDINGS=2, base_seed=42):
    recordings = []
    for i in range(NUM_RECORDINGS):
        recording, _ = generate_ground_truth_recording(num_channels=8, durations=[10.0], seed=base_seed + i)

        if i % 2 == 0:
            recording.set_channel_groups(["0"] * 4 + ["1"] * 4)
        else:
            recording.set_channel_groups([0] * 4 + [1] * 4)
        recordings.append(recording)
    return recordings


def get_job_list(base_folder):
    jobs = []
    recordings = create_recordings(NUM_RECORDINGS)
    for i, recording in enumerate(recordings):
        for sorter_name in SORTERS:
            kwargs = dict(
                sorter_name=sorter_name,
                recording=recording,
                folder=base_folder / f"{sorter_name}_rec{i}",
                verbose=True,
                raise_error=False,
            )
            jobs.append(kwargs)

    return jobs


@pytest.fixture(scope="function")
def job_list(create_cache_folder):
    cache_folder = create_cache_folder
    folder = cache_folder / "sorting_output"
    return get_job_list(folder)


def test_run_sorter_jobs_loop(job_list):
    sortings = run_sorter_jobs(job_list, engine="loop", return_output=True)
    print(sortings)


@pytest.mark.skipif(True, reason="tridesclous is already multiprocessing, joblib cannot run it in parralel")
def test_run_sorter_jobs_joblib(job_list):
    sortings = run_sorter_jobs(
        job_list, engine="joblib", engine_kwargs=dict(n_jobs=2, backend="loky"), return_output=True
    )
    print(sortings)


@pytest.mark.skipif(
    True, reason="tridesclous is already multiprocessing, processpoolexecutor cannot run it in parralel"
)
def test_run_sorter_jobs_processpoolexecutor(job_list, create_cache_folder):
    cache_folder = create_cache_folder
    if (cache_folder / "sorting_output").is_dir():
        shutil.rmtree(cache_folder / "sorting_output")
    sortings = run_sorter_jobs(
        job_list, engine="processpoolexecutor", engine_kwargs=dict(max_workers=2), return_output=True
    )
    print(sortings)


@pytest.mark.skipif(True, reason="This is tested locally")
def test_run_sorter_jobs_dask(job_list):

    # create a dask Client for a slurm queue
    from dask.distributed import Client

    test_mode = "local"
    # test_mode = "client_slurm"

    if test_mode == "local":
        client = Client()
    elif test_mode == "client_slurm":
        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(
            processes=1,
            cores=1,
            memory="12GB",
            python=sys.executable,
            walltime="12:00:00",
        )
        cluster.scale(2)
        client = Client(cluster)

    # dask
    t0 = time.perf_counter()
    run_sorter_jobs(job_list, engine="dask", engine_kwargs=dict(client=client))
    t1 = time.perf_counter()
    print(t1 - t0)


@pytest.mark.skip("Slurm launcher need a machine with slurm")
def test_run_sorter_jobs_slurm(job_list, create_cache_folder):
    cache_folder = create_cache_folder

    working_folder = cache_folder / "test_run_SORTERS_slurm"
    if working_folder.is_dir():
        shutil.rmtree(working_folder)

    tmp_script_folder = working_folder / "slurm_scripts"

    run_sorter_jobs(
        job_list,
        engine="slurm",
        engine_kwargs=dict(
            tmp_script_folder=tmp_script_folder,
            cpus_per_task=32,
            mem="32G",
        ),
    )


@pytest.mark.skipif(system() != "Linux", reason="Assumes we are on Linux to run SLURM")
def test_run_sorter_jobs_slurm_kwargs(mocker, tmp_path, job_list):
    """
    Mock `subprocess.run()` to check that engine_kwargs are
    propagated to the call as expected.
    """
    # First, mock `subprocess.run()`, set up a call to `run_sorter_jobs`
    # then check the mocked `subprocess.run()` was called with the
    # expected signature. Two jobs are passed in `jobs_list`, first
    # check the most recent call.
    mock_subprocess_run = mocker.patch("spikeinterface.sorters.launcher.subprocess.run")

    tmp_script_folder = tmp_path / "slurm_scripts"

    engine_kwargs = dict(
        tmp_script_folder=tmp_script_folder,
        sbatch_args={
            "cpus-per-task": 32,
            "mem": "32G",
            "gres": "gpu:1",
            "any_random_kwarg": 12322,
        },
    )

    run_sorter_jobs(job_list, engine="slurm", engine_kwargs=engine_kwargs)

    script_0_path = f"{tmp_script_folder}/si_script_0.py"
    script_1_path = f"{tmp_script_folder}/si_script_1.py"

    expected_command = [
        "sbatch",
        "--cpus-per-task",
        "32",
        "--mem",
        "32G",
        "--gres",
        "gpu:1",
        "--any_random_kwarg",
        "12322",
        script_1_path,
    ]
    mock_subprocess_run.assert_called_with(expected_command, capture_output=True, text=True)

    # Next, check the fisrt call (which sets up `si_script_0.py`)
    # also has the expected arguments.
    expected_command[9] = script_0_path
    assert mock_subprocess_run.call_args_list[0].args[0] == expected_command

    # Next, check that defaults are used properly when no kwargs are
    # passed. This will default to `_default_engine_kwargs` as
    # set in `launcher.py`
    run_sorter_jobs(
        job_list,
        engine="slurm",
        engine_kwargs={"tmp_script_folder": tmp_script_folder},
    )
    expected_command = ["sbatch", "--cpus-per-task", "1", "--mem", "1G", script_1_path]
    mock_subprocess_run.assert_called_with(expected_command, capture_output=True, text=True)

    # Finally, check that the `tmp_script_folder` is generated on the
    # fly as expected. A random foldername is generated, just check that
    # the folder to which the scripts are saved is in the `tempfile` format.
    run_sorter_jobs(
        job_list,
        engine="slurm",
        engine_kwargs=None,
    )
    tmp_script_folder = Path(tempfile.gettempdir()) / "spikeinterface_slurm_"
    assert str(tmp_script_folder) in mock_subprocess_run.call_args_list[-1].args[0][5]


def test_run_sorter_by_property(create_cache_folder):
    cache_folder = create_cache_folder
    working_folder1 = cache_folder / "test_run_sorter_by_property_1"
    if working_folder1.is_dir():
        shutil.rmtree(working_folder1)

    working_folder2 = cache_folder / "test_run_sorter_by_property_2"
    if working_folder2.is_dir():
        shutil.rmtree(working_folder2)

    recordings = create_recordings(NUM_RECORDINGS)

    rec0 = recordings[0]
    rec0_by = rec0.split_by("group")
    group_names0 = list(rec0_by.keys())

    sorter_name = "tridesclous2"
    sorting0 = run_sorter_by_property(sorter_name, rec0, "group", working_folder1, engine="loop", verbose=False)
    assert "group" in sorting0.get_property_keys()
    assert all([g in group_names0 for g in sorting0.get_property("group")])

    rec1 = recordings[1]
    rec1_by = rec1.split_by("group")
    group_names1 = list(rec1_by.keys())

    sorter_name = "tridesclous2"
    sorting1 = run_sorter_by_property(sorter_name, rec1, "group", working_folder2, engine="loop", verbose=False)
    assert "group" in sorting1.get_property_keys()
    assert all([g in group_names1 for g in sorting1.get_property("group")])


if __name__ == "__main__":
    # setup_module()
    tmp_folder = Path("tmp")
    job_list = get_job_list(tmp_folder)

    # test_run_sorter_jobs_loop(job_list)
    # test_run_sorter_jobs_joblib(job_list)
    # test_run_sorter_jobs_processpoolexecutor(job_list)
    # test_run_sorter_jobs_multiprocessing(job_list)
    # test_run_sorter_jobs_dask(job_list)
    # test_run_sorter_jobs_slurm(job_list)

    test_run_sorter_by_property(tmp_folder)
