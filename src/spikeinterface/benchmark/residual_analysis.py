from spikeinterface.core.generate import InjectTemplatesRecording


def analyse_residual(
    analyzer,
    detect_peaks_kwargs=dict(
        method="locally_exclusive",
        peak_sign="both",
        detect_threshold=6.0,
    ),
    **job_kwargs,
):
    """
    This create the residual by removing each spike from the recording.
    This take in account the spike amplitude scaling, analyzer need "amplitude_scalings" extensions.
    Then a peak detector is run on this residual tarces and then number of peaks can be analyzed (the less the better).

    This residual is not perfect at the moement because it do not take in the account the jitter per spikes
    and so the residual can be high for high amplitude when there is a inherent jitter per spike.

    Paramters
    ----------
    analyzer : SortingAnalyzer

    Returns
    -------
    residual : Recording
        The resdiual
    peaks : np.array
        The peaks vector detected on the residual.

    """
    from spikeinterface.sortingcomponents.peak_detection import detect_peaks

    residual = make_residual_recording(analyzer)

    peaks = detect_peaks(residual, **detect_peaks_kwargs, **job_kwargs)
    energies = compute_energy(residual, **job_kwargs)

    return residual, peaks, energies


def make_residual_recording(analyzer):
    """
    This make a lazy recording residual from an anlyzer.

    Paramters
    ----------
    analyzer : SortingAnalyzer

    Returns
    -------
    residual : Recording
        The resdiual
    """

    templates = analyzer.get_extension("templates").get_templates(outputs="Templates")
    neg_templates_array = templates.templates_array.copy()
    neg_templates_array *= -1

    amplitude_factor = analyzer.get_extension("amplitude_scalings").get_data()

    residual = InjectTemplatesRecording(
        analyzer.sorting,
        neg_templates_array,
        nbefore=templates.nbefore,
        parent_recording=analyzer.recording,
        amplitude_factor=amplitude_factor,
    )
    residual.name = "ResidualRecording"

    return residual


from spikeinterface.core.node_pipeline import PeakDetector
import numpy as np

base_peak_dtype = [
    ("sample_index", "int64"),
    ("channel_index", "int64"),
    ("energy", "float64"),
    ("segment_index", "int64"),
]

class ComputeEnergy(PeakDetector):

    name = "energy"
    preferred_mp_context = None

    def __init__(
        self,
        recording
    ):
        PeakDetector.__init__(self, recording, return_output=True)

    def get_trace_margin(self):
        return 0

    def get_dtype(self):
        return self._dtype

    def compute(self, traces, start_frame, end_frame, segment_index, max_margin):
        num_channels = traces.shape[1]
        energy = np.zeros(num_channels, dtype=base_peak_dtype)
        energy["sample_index"] = 0
        energy["segment_index"] = segment_index
        energy["channel_index"] = range(num_channels)
        energy["energy"] = np.linalg.norm(traces, axis=0)/np.sqrt(traces.shape[0])
        return (energy,)


def compute_energy(recording, job_kwargs=dict()):

    from spikeinterface.core.node_pipeline import (
        run_node_pipeline,
    )
    from spikeinterface.core.job_tools import fix_job_kwargs

    job_kwargs = fix_job_kwargs(job_kwargs)

    node0 = ComputeEnergy(
        recording
    )

    residuals = run_node_pipeline(
        recording,
        [node0],
        job_kwargs,
        job_name="compute energy",
    )
    return residuals