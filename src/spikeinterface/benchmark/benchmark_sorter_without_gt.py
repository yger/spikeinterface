"""
This replace the previous `GroundTruthStudy`
"""

import numpy as np
from spikeinterface.core import NumpySorting, create_sorting_analyzer
from .benchmark_base import Benchmark, BenchmarkStudy, MixinStudyUnitCount
from spikeinterface.sorters import run_sorter
from spikeinterface.comparison import compare_multiple_sorters

from spikeinterface.benchmark import analyse_residual


# TODO later integrate CollisionGTComparison optionally in this class.


class SorterBenchmarkWithoutGroundTruth(Benchmark):
    def __init__(self, recording, gt_sorting, params, sorter_folder):
        self.recording = recording
        self.gt_sorting = gt_sorting
        self.params = params
        self.sorter_folder = sorter_folder
        self.result = {}

    def run(self):
        # run one sorter sorter_name is must be in params
        raw_sorting = run_sorter(recording=self.recording, folder=self.sorter_folder, **self.params)
        sorting = NumpySorting.from_sorting(raw_sorting)
        self.result = {"sorting": sorting}

    def compute_result(self, residulal_peak_threshold=6, **job_kwargs):

        sorting = self.result["sorting"]
        analyzer = create_sorting_analyzer(sorting, self.recording, sparse=True, format="memory", **job_kwargs)
        analyzer.compute("random_spikes")
        analyzer.compute("templates")
        analyzer.compute("noise_levels")
        analyzer.compute({"spike_amplitudes": {}, "amplitude_scalings": {"handle_collisions": False}}, **job_kwargs)

        analyzer.compute("quality_metrics", **job_kwargs)

        residual, peaks, energies = analyse_residual(
            analyzer,
            detect_peaks_kwargs=dict(
                method="locally_exclusive",
                peak_sign="neg",
                detect_threshold=residulal_peak_threshold,
            ),
            **job_kwargs,
        )

        self.result["residual"] = residual
        self.result["sorter_analyzer"] = analyzer
        self.result["peaks_from_residual"] = peaks
        self.result["energies_from_residual"] = energies

    _run_key_saved = [
        ("sorting", "sorting"),
    ]
    _result_key_saved = [
        # note that this multi_comp is the same accros benchmark (cases)
        ("multi_comp", "pickle"),
        ("residual", "pickle"),
        ("sorter_analyzer", "sorting_analyzer"),
        ("peaks_from_residual", "npy"),
        ("energies_from_residual", "npy"),
    ]


class SorterStudyWithoutGroundTruth(BenchmarkStudy):
    """
    This class is an alternative to SorterStudy when the dataset do not have groundtruth.

    This is mainly base on the residual analysis.
    """

    benchmark_class = SorterBenchmarkWithoutGroundTruth

    def create_benchmark(self, key):
        dataset_key = self.cases[key]["dataset"]
        recording, gt_sorting = self.datasets[dataset_key]
        params = self.cases[key]["params"]
        sorter_folder = self.folder / "sorters" / self.key_to_str(key)
        benchmark = SorterBenchmarkWithoutGroundTruth(recording, gt_sorting, params, sorter_folder)
        return benchmark

    def _get_comparison_groups(self):
        # multicomparison are done on all cases sharing the same dataset key.
        case_keys = list(self.cases.keys())
        groups = {}
        for case_key in case_keys:
            data_key = self.cases[case_key]["dataset"]
            if data_key not in groups:
                groups[data_key] = []
            groups[data_key].append(case_key)
        return groups

    def compute_results(
        self, case_keys=None, verbose=False, delta_time=0.4, match_score=0.5, chance_score=0.1, **result_params
    ):
        # Here we need a hack because the results is not computed case by case but all at once

        assert case_keys is None, "SorterStudyWithoutGroundTruth do not permit compute_results for sub cases"

        # allways the full list
        case_keys = list(self.cases.keys())

        # First : this do the case by case internally SorterBenchmarkWithoutGroundTruth.compute_result()
        BenchmarkStudy.compute_results(self, case_keys=case_keys, verbose=verbose, **result_params)

        # Then we need to compute the multicomparison for case that have the same dataset key.
        groups = self._get_comparison_groups()

        for data_key, group in groups.items():

            sorting_list = [self.get_result(key)["sorting"] for key in group]
            name_list = [key for key in group]
            multi_comp = compare_multiple_sorters(
                sorting_list,
                name_list=name_list,
                delta_time=delta_time,
                match_score=0.5,
                chance_score=0.1,
                agreement_method="count",
                n_jobs=-1,
                spiketrain_mode="union",
                verbose=verbose,
                do_matching=True,
            )
            # and then the same multi comp is stored for each case_key
            for key in case_keys:
                benchmark = self.benchmarks[key]
                benchmark.result["multi_comp"] = multi_comp
                benchmark.save_result(self.folder / "results" / self.key_to_str(key))

    def plot_residual_peak_amplitudes(self, figsize=None, num_bins=50):
        import matplotlib.pyplot as plt

        groups = self._get_comparison_groups()
        colors = self.get_colors()

        for data_key, group in groups.items():
            fig, axes = plt.subplots(1, 3, figsize=figsize)

            lim0, lim1 = np.inf, -np.inf

            for key in group:
                peaks = self.get_result(key)["peaks_from_residual"]

                lim0 = min(lim0, np.min(peaks["amplitude"]))
                lim1 = max(lim1, np.max(peaks["amplitude"]))

            bins = np.linspace(lim0, lim1, num_bins)
            if lim1 < 0:
                lim1 = 0
            if lim0 > 0:
                lim0 = 0
            
            bins_channel = range(self.get_result(key)["sorter_analyzer"].recording.get_num_channels())

            for idx, key in enumerate(group):
                peaks = self.get_result(key)["peaks_from_residual"]
                count, _ = np.histogram(peaks["amplitude"], bins=bins)
                axes[0].plot(bins[:-1], count, color=colors[key], label=self.cases[key]["label"])

                axes[1].bar([idx], np.sum(count)/(lim1 - lim0), color=colors[key], label=self.cases[key]["label"])

                count, _ = np.histogram(peaks["channel_index"], bins=bins_channel)
                axes[2].plot(bins_channel[:-1], count, color=colors[key], label=self.cases[key]["label"])

            axes[0].set_title("Residual peak amplitudes")
            axes[0].set_xlabel("Amplitude")
            axes[0].set_ylabel("Count")
            axes[1].set_ylabel('Area under curve')
            axes[2].set_title("Residual peak channel index")
            axes[2].set_xlabel("Channel index")
            axes[2].set_ylabel("Count")
            axes[0].legend()
            plt.tight_layout()
    
    def plot_residual_energies(self, figsize=None, num_bins=50, case_keys=None, levels_to_group_by=None):
        import matplotlib.pyplot as plt

        groups = self._get_comparison_groups()
        colors = self.get_colors()

        for data_key, group in groups.items():
            fig, axes = plt.subplots(1, 2, figsize=figsize)

            lim0, lim1 = np.inf, -np.inf

            for key in group:
                peaks = self.get_result(key)["energies_from_residual"]
                lim0 = min(lim0, np.min(peaks["energy"]))
                lim1 = max(lim1, np.max(peaks["energy"]))

            bins = np.linspace(lim0, lim1, num_bins)
            if lim1 < 0:
                lim1 = 0
            if lim0 > 0:
                lim0 = 0
            
            for key in group:
                peaks = self.get_result(key)["energies_from_residual"]
                count, _ = np.histogram(peaks["energy"], bins=bins)
                axes[0].plot(bins[:-1], count, color=colors[key], label=self.cases[key]["label"])

            axes[0].set_title("Residual energies")
            axes[0].set_xlabel("Energy")
            axes[0].set_ylabel("Count")
            axes[0].set_yscale('log')
            axes[0].legend()

        case_keys = None
        if case_keys is None:
            case_keys = list(self.cases.keys())

        data = []

        case_keys, _ = self.get_grouped_keys_mapping(levels_to_group_by=levels_to_group_by, case_keys=case_keys)
        labels = []
        for i, key1 in enumerate(case_keys):
            for j, key2 in enumerate(case_keys):
                if i < j:
                    labels += [f"{self.cases[key1]['label']}/{self.cases[key2]['label']}"]
                    data_1 = self.get_result(key1)['energies_from_residual']["energy"]
                    data_2 = self.get_result(key2)['energies_from_residual']["energy"]
                    data += [data_1 / data_2]    
                    
        axes[1].violinplot(data, showmeans=False, showmedians=True)
        axes[1].set_xticks(np.arange(len(data)) + 1, labels, rotation=45)

    def plot_quality_metrics_comparison_on_agreement(self, qm_name='rp_contamination', 
                                                     case_keys=None, levels_to_group_by=None,figsize=None):
        import matplotlib.pyplot as plt

        groups = self._get_comparison_groups()

        if case_keys is None:
            case_keys = list(self.cases.keys())
        case_keys, labels = self.get_grouped_keys_mapping(levels_to_group_by=levels_to_group_by, case_keys=case_keys)

        from .benchmark_plot_tools import despine
        
        num_methods = len(case_keys)
        for data_key, group in groups.items():
            n = len(group)
            fig, axs = plt.subplots(ncols=n - 1, nrows=n - 1, figsize=figsize, squeeze=False)
            for i, key1 in enumerate(group):
                for j, key2 in enumerate(group):
                    if i < j:
                        ax = axs[i, j - 1]
                        label1 = self.cases[key1]['label']
                        label2 = self.cases[key2]['label']

                        if i == j - 1:
                            ax.set_xlabel(label2)
                            ax.set_ylabel(label1)

                        multi_comp = self.get_result(key1)['multi_comp']
                        comp = multi_comp.comparisons[key1, key2]

                        match_12 = comp.hungarian_match_12
                        if match_12.dtype.kind =='i':
                            mask = match_12.values != -1
                        if match_12.dtype.kind =='U':
                            mask = match_12.values != ''

                        common_unit1_ids = match_12[mask].index
                        common_unit2_ids = match_12[mask].values
                        metrics1 = self.get_result(key1)["sorter_analyzer"].get_extension("quality_metrics").get_data()
                        metrics2 = self.get_result(key2)["sorter_analyzer"].get_extension("quality_metrics").get_data()

                        values1 = metrics1.loc[common_unit1_ids, qm_name].values
                        values2 = metrics2.loc[common_unit2_ids, qm_name].values

                        ax.scatter(values1, values2)
                        despine(ax)
                        if i != j - 1:
                            ax.set_xlabel("")
                            ax.set_ylabel("")
                            ax.set_xticks([])
                            ax.set_yticks([])
                            ax.set_xticklabels([])
                            ax.set_yticklabels([])
                    else:
                        if j >= 1 and i < num_methods - 1:
                            ax = axs[i, j - 1]
                            ax.axis("off")
