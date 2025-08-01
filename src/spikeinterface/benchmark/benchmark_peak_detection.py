from __future__ import annotations

from spikeinterface.sortingcomponents.peak_detection import detect_peaks
from spikeinterface.core import NumpySorting
from spikeinterface.comparison import GroundTruthComparison
from spikeinterface.widgets import (
    plot_agreement_matrix,
)
from spikeinterface.comparison.comparisontools import make_matching_events
from spikeinterface.core import get_noise_levels
from spikeinterface.benchmark.benchmark_plot_tools import despine

import numpy as np
from spikeinterface.core.job_tools import fix_job_kwargs, split_job_kwargs
from .benchmark_base import Benchmark, BenchmarkStudy
from spikeinterface.core.basesorting import minimum_spike_dtype
from spikeinterface.core.sortinganalyzer import create_sorting_analyzer
from .benchmark_plot_tools import fit_sigmoid, sigmoid


class PeakDetectionBenchmark(Benchmark):

    def __init__(self, recording, gt_sorting, params, gt_peaks, exhaustive_gt=True, delta_t_ms=0.2):
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.gt_peaks = gt_peaks
        self.params = params
        self.exhaustive_gt = exhaustive_gt
        assert "method" in self.params, "Method should be specified in the params!"
        self.method = self.params.get("method")
        self.delta_frames = int(delta_t_ms * self.recording.sampling_frequency / 1000)
        self.params = self.params["method_kwargs"]
        self.result = {}

    def run(self, **job_kwargs):
        peaks = detect_peaks(self.recording, self.method, **self.params, **job_kwargs)
        self.result["peaks"] = peaks

    def compute_result(self, **result_params):
        result_params, job_kwargs = split_job_kwargs(result_params)
        job_kwargs = fix_job_kwargs(job_kwargs)
        sorting_analyzer = create_sorting_analyzer(
            self.gt_sorting, self.recording, format="memory", sparse=False, **job_kwargs
        )
        sorting_analyzer.compute("random_spikes")
        sorting_analyzer.compute("templates", **job_kwargs)
        sorting_analyzer.compute("spike_amplitudes", **job_kwargs)
        self.result["gt_amplitudes"] = sorting_analyzer.get_extension("spike_amplitudes").get_data()
        self.result["gt_templates"] = sorting_analyzer.get_extension("templates").get_data()

        spikes = self.result["peaks"]
        self.result["peak_on_channels"] = NumpySorting.from_peaks(
            spikes, self.recording.sampling_frequency, unit_ids=self.recording.channel_ids
        )
        spikes = self.gt_peaks
        self.result["gt_on_channels"] = NumpySorting.from_peaks(
            spikes, self.recording.sampling_frequency, unit_ids=self.recording.channel_ids
        )

        self.result["gt_comparison"] = GroundTruthComparison(
            self.result["gt_on_channels"], self.result["peak_on_channels"], exhaustive_gt=self.exhaustive_gt
        )

        gt_peaks = self.gt_sorting.to_spike_vector()
        peaks = self.result["peaks"]
        times1 = peaks["sample_index"]
        times2 = spikes["sample_index"]

        print("The gt recording has {} peaks and {} have been detected".format(len(times1), len(times2)))

        matches = make_matching_events(times1, times2, self.delta_frames)
        gt_matches = matches["index2"]
        detected_matches = matches["index1"]

        self.result["matches"] = {"deltas": matches["delta_frame"]}
        self.result["matches"]["labels"] = gt_peaks["unit_index"][gt_matches]
        self.result["matches"]["channels"] = spikes["unit_index"][gt_matches]

        sorting = np.zeros(gt_matches.size, dtype=minimum_spike_dtype)
        sorting["sample_index"] = peaks[detected_matches]["sample_index"]
        sorting["unit_index"] = gt_peaks["unit_index"][gt_matches]
        sorting["segment_index"] = peaks[detected_matches]["segment_index"]
        order = np.lexsort((sorting["sample_index"], sorting["segment_index"]))
        sorting = sorting[order]
        self.result["sliced_gt_sorting"] = NumpySorting(
            sorting, self.recording.sampling_frequency, self.gt_sorting.unit_ids
        )
        self.result["sliced_gt_comparison"] = GroundTruthComparison(
            self.gt_sorting, self.result["sliced_gt_sorting"], exhaustive_gt=self.exhaustive_gt
        )

        ratio = 100 * len(gt_matches) / len(times2)
        print("Only {0:.2f}% of gt peaks are matched to detected peaks".format(ratio))

        sorting_analyzer = create_sorting_analyzer(
            self.result["sliced_gt_sorting"], self.recording, format="memory", sparse=False, **job_kwargs
        )
        sorting_analyzer.compute("random_spikes")
        sorting_analyzer.compute("templates", **job_kwargs)

        self.result["templates"] = sorting_analyzer.get_extension("templates").get_data()

    _run_key_saved = [("peaks", "npy")]

    _result_key_saved = [
        ("gt_comparison", "pickle"),
        ("sliced_gt_sorting", "sorting"),
        ("sliced_gt_comparison", "pickle"),
        ("sliced_gt_sorting", "sorting"),
        ("peak_on_channels", "sorting"),
        ("gt_on_channels", "sorting"),
        ("matches", "pickle"),
        ("templates", "npy"),
        ("gt_amplitudes", "npy"),
        ("gt_templates", "npy"),
    ]


class PeakDetectionStudy(BenchmarkStudy):

    benchmark_class = PeakDetectionBenchmark

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        init_kwargs = self.cases[key]["init_kwargs"]
        benchmark = PeakDetectionBenchmark(recording, gt_sorting, params, **init_kwargs)
        return benchmark

    def plot_agreements_by_channels(self, case_keys=None, figsize=(15, 15)):
        if case_keys is None:
            case_keys = list(self.cases.keys())
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):
            ax = axs[0, count]
            ax.set_title(self.cases[key]["label"])
            plot_agreement_matrix(self.get_result(key)["gt_comparison"], ax=ax)

    def plot_agreements_by_units(self, case_keys=None, figsize=(15, 15)):
        if case_keys is None:
            case_keys = list(self.cases.keys())
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)

        for count, key in enumerate(case_keys):
            ax = axs[0, count]
            ax.set_title(self.cases[key]["label"])
            plot_agreement_matrix(self.get_result(key)["sliced_gt_comparison"], ax=ax)

    def plot_detected_amplitudes(self, case_keys=None, figsize=(15, 5), detect_threshold=None, axs=None):

        if case_keys is None:
            case_keys = list(self.cases.keys())
        import matplotlib.pyplot as plt

        if axs is None:
            fig, axs = plt.subplots(ncols=len(case_keys), figsize=figsize, squeeze=False)
        else:
            fig = axs[0].get_figure()
            assert len(axs) == len(case_keys), "axs should be the same length as case_keys"

        for count, key in enumerate(case_keys):
            ax = axs[count]
            despine(ax)
            data1 = self.get_result(key)["peaks"]["amplitude"]
            data2 = self.get_result(key)["gt_amplitudes"]
            color = self.get_colors()[key]
            bins = np.linspace(data2.min(), data2.max(), 100)
            ax.hist(data1, bins=bins, label="detected", histtype="step", color=color, linewidth=2)
            ax.hist(data2, bins=bins, alpha=0.1, label="gt", color="k")
            ax.set_yscale("log")
            # ax.set_title(self.cases[key]["label"])
            ax.legend()
            if detect_threshold is not None:
                noise_levels = get_noise_levels(self.benchmarks[key].recording, return_in_uV=False).mean()
                ymin, ymax = ax.get_ylim()
                abs_threshold = -detect_threshold * noise_levels
                ax.plot([abs_threshold, abs_threshold], [ymin, ymax], "k--")

        return fig

    def plot_deltas_per_cells(self, case_keys=None, figsize=(15, 5)):

        if case_keys is None:
            case_keys = list(self.cases.keys())
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(ncols=len(case_keys), nrows=1, figsize=figsize, squeeze=False)
        for count, key in enumerate(case_keys):
            ax = axs[0, count]
            gt_sorting = self.benchmarks[key].gt_sorting
            data = self.get_result(key)["matches"]
            for unit_ind, unit_id in enumerate(gt_sorting.unit_ids):
                mask = data["labels"] == unit_id
                ax.violinplot(
                    data["deltas"][mask], [unit_ind], widths=2, showmeans=True, showmedians=False, showextrema=False
                )
            ax.set_title(self.cases[key]["label"])
            ax.set_xticks(np.arange(len(gt_sorting.unit_ids)), gt_sorting.unit_ids)
            ax.set_ylabel("# frames")
            ax.set_xlabel("unit id")

    def plot_mean_deltas(self, case_keys=None, figsize=(15, 5), ax=None):

        if case_keys is None:
            case_keys = list(self.cases.keys())
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(1, figsize=figsize)
        else:
            fig = ax.get_figure()

        results = {}
        labels = []
        colors = []
        for count, key in enumerate(case_keys):
            gt_sorting = self.benchmarks[key].gt_sorting
            results[key] = []
            labels += [self.cases[key]["label"]]
            data = self.get_result(key)["matches"]
            for unit_ind, unit_id in enumerate(gt_sorting.unit_ids):
                mask = data["labels"] == unit_id
                results[key] += [np.mean(data["deltas"][mask])]

            colors += [self.get_colors()[key]]
        despine(ax)
        plots = ax.violinplot(
            results.values(),
            range(len(case_keys)),
            showmeans=True,
            showmedians=False,
            showextrema=False,
        )

        # Set the color of the violin patches
        for pc, color in zip(plots["bodies"], colors):
            pc.set_facecolor(color)
            pc.set_edgecolor(color)

        plots["cmeans"].set_colors(colors)

        # ax.set_title(self.cases[key]["label"])
        ax.set_xticks(range(len(case_keys)), labels, rotation=45)
        ax.set_ylabel("# frames")
        # ax.set_xlabel("unit id")

    def plot_template_similarities(self, case_keys=None, metric="l2", figsize=(15, 5), detect_threshold=None, ax=None):

        if case_keys is None:
            case_keys = list(self.cases.keys())
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=figsize, squeeze=True)
        else:
            fig = ax.get_figure()

        for key in case_keys:

            import sklearn.metrics

            gt_templates = self.get_result(key)["gt_templates"]
            found_templates = self.get_result(key)["templates"]
            num_templates = len(gt_templates)
            distances = np.zeros(num_templates)

            for i in range(num_templates):

                a = gt_templates[i].flatten()
                b = found_templates[i].flatten()

                if metric == "cosine":
                    import sklearn.metrics

                    distances[i] = sklearn.metrics.pairwise.cosine_similarity(a[None, :], b[None, :])[0, 0]
                else:
                    distances[i] = sklearn.metrics.pairwise_distances(a[None, :], b[None, :], metric)[0, 0]

            color = self.get_colors()[key]

            label = self.cases[key]["label"]
            analyzer = self.get_sorting_analyzer(key)
            metrics = analyzer.get_extension("quality_metrics").get_data()
            x = metrics["snr"].values
            y = distances
            ax.scatter(x, y, marker=".", label=label, color=color)
            despine(ax)
            popt = fit_sigmoid(x, y, p0=None)
            xfit = np.linspace(0, max(metrics["snr"].values), 100)
            ax.plot(xfit, sigmoid(xfit, *popt), color=color)

        if detect_threshold is not None:
            ymin, ymax = ax.get_ylim()
            ax.plot([detect_threshold, detect_threshold], [ymin, ymax], "k--")

        ax.legend()
        ax.set_xlabel("snr")
        ax.set_ylabel(metric)
        return fig
